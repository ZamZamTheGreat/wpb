import os
import re
import json
import uuid
import redis
import openai
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, abort
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_session import Session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask import make_response
from openai import OpenAI
import requests
from markupsafe import escape

app = Flask(__name__)
app.jinja_env.globals['now'] = datetime.now
app.config['SESSION_COOKIE_SECURE'] = True

# ─── Setup ───────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Paths & Uploads ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads', 'VisionaryAutomation')

# ─── Flask app config ───────────────────────────────────────────────────────
app.secret_key = 'super-secret-key'  # or replace with a stronger one
# ─── Paths & Uploads ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Persistent Disk base path (mounted at /var/data in Render)
PERSISTENT_DISK_PATH = "/var/data"
os.makedirs(PERSISTENT_DISK_PATH, exist_ok=True)

# Flask configuration
app.config['UPLOAD_FOLDER'] = PERSISTENT_DISK_PATH  # base folder for all agent data
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt', 'md'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max per file

# --- Load Redis from .env ---

REDIS_URL = os.getenv("SESSION_REDIS_URL")  # e.g. redis://default:<password>@<host>:6379

try:
    # Use decode_responses=False for binary-safe session storage
    r = redis.from_url(REDIS_URL, decode_responses=False)
    r.ping()
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_REDIS'] = r
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    Session(app)
    print("✅ Using Redis for sessions")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}. Using filesystem sessions")
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_REDIS'] = None
Session(app)

# ─── OpenAI API Key ─────────────────────────────────────────────────────────
openai.api_key = os.getenv('OPENAI_API_KEY')

# ─── Login ───────────────────────────────────────────────────────────────────
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id_): self.id = id_
    @property
    def is_admin(self): return self.id == os.getenv('ADMIN_USERNAME')
    
@login_manager.user_loader
def load_user(uid):
    return User(uid) if uid == os.getenv('ADMIN_USERNAME') else None

# ─── Utilities ───────────────────────────────────────────────────────────────

DOC_JSON_PATH = os.path.join(BASE_DIR, "agent_docs.json")

def load_prompt(name):
    """Load system prompt text for a given agent."""
    prompt_path = os.path.join(BASE_DIR, 'prompts', f'{name}.txt')
    if os.path.exists(prompt_path):
        with open(prompt_path, encoding='utf-8') as f:
            return f.read()
    return ""

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_user_upload_dir(agent_id):
    """Return the agent-specific folder under /var/data and ensure it exists."""
    path = os.path.join('/var/data', agent_id)
    os.makedirs(path, exist_ok=True)
    return path

def load_global_docs():
    """Load the JSON registry of uploaded files."""
    if os.path.exists(DOC_JSON_PATH):
        with open(DOC_JSON_PATH, 'r', encoding='utf-8') as f:
            docs = json.load(f)
            # Normalize old paths: strip agent folder if mistakenly included
            for agent_id, paths in docs.items():
                if not isinstance(paths, list):
                    paths = [paths]
                normalized = [os.path.basename(p) for p in paths]
                docs[agent_id] = normalized
            return docs
    return {}

def save_global_docs(docs):
    """Save the JSON registry of uploaded files."""
    with open(DOC_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2)

def preload_documents():
    """
    Deprecated: use load_agent_documents() instead.
    Can still load all documents from disk regardless of JSON registry.
    """
    all_docs = {}
    base_dir = '/var/data'  # base folder for all agent uploads

    if not os.path.exists(base_dir):
        return all_docs

    for agent_id in os.listdir(base_dir):
        agent_dir = os.path.join(base_dir, agent_id)
        if not os.path.isdir(agent_dir):
            continue
        for filename in os.listdir(agent_dir):
            if not allowed_file(filename):
                continue
            file_path = os.path.join(agent_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                all_docs.setdefault(agent_id, []).append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return all_docs

# Global dictionary to hold loaded docs content
AGENT_DOCUMENTS = {}
def load_agent_documents():
    """
    Loads all agent documents from /var/data/<agent_id>/ and stores
    their content in AGENT_DOCUMENTS, keyed by agent_id.
    """
    global AGENT_DOCUMENTS
    AGENT_DOCUMENTS = {}

    global_docs = load_global_docs()  # JSON registry of uploaded files

    for agent_id, file_names in global_docs.items():
        if not isinstance(file_names, list):
            file_names = [file_names]  # ensure list

        contents = []
        agent_folder = os.path.join('/var/data', agent_id)  # agent-specific folder
        os.makedirs(agent_folder, exist_ok=True)  # ensure folder exists

        for file_name in file_names:
            # Only join with agent_folder, no double agent folder
            full_path = os.path.normpath(os.path.join(agent_folder, file_name))

            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        contents.append(f.read())
                except UnicodeDecodeError:
                    print(f"Skipping non-text file for agent '{agent_id}': {full_path}")
                except Exception as e:
                    print(f"Failed to load document for agent '{agent_id}' at {full_path}: {e}")
            else:
                print(f"Document path does not exist for agent '{agent_id}': {full_path}")

        AGENT_DOCUMENTS[agent_id] = contents


def chat_url_for(agent_name: str) -> str:
    """Absolute link to an agent's chat page."""
    try:
        # Works inside a request context; builds https/http + host automatically
        return url_for('chat', agent_id=agent_name, _external=True)
    except RuntimeError:
        # Fallback if somehow called without request context
        return f"/chat/{agent_name}"
    
# ─── Agents ──────────────────────────────────────────────────────────────────
AGENT_CONFIG = {
    'Ruanca-AI': {'system_prompt':load_prompt('Ruanca')},
    'Carl-AI': {'system_prompt':load_prompt('Carl')},
    'Natalie-AI': {'system_prompt': load_prompt('Natalie')},
    'Deborah-AI': {'system_prompt': load_prompt('Deborah')},
    'Search-AI': {'system_prompt': load_prompt('Search')},
    'Head of property-AI': {'system_prompt': load_prompt('Head of Property')},
}
global_docs = load_global_docs()
AGENT_DOCUMENTS = preload_documents()

# Mapping agents to their Tally form links
TALLY_FORMS = {
    "Ruanca-AI": "https://tally.so/r/mD9W4j",
    "Carl-AI": "https://tally.so/r/mZr85a",
    "Natalie-AI": "https://tally.so/r/3XPLld",
    "Deborah-AI": "https://tally.so/r/3XPLld",
    # Add more agents here as needed
}

# ─── Chat Handling ───────────────────────────────────────────────────────────
def user_agent_data(agent_id):
    session.setdefault('agent_data', {})
    session['agent_data'].setdefault(agent_id, {
        'history': [], 'rag_file': None, 'document_name': []
    })

    if not session['agent_data'][agent_id]['rag_file'] and agent_id in global_docs:
        docs = global_docs[agent_id]

        if isinstance(docs, list):
            # Join relative paths with UPLOAD_FOLDER to get full absolute paths
            full_paths = [os.path.join(app.config['UPLOAD_FOLDER'], d) for d in docs]

            # Keep only those that exist on disk
            existing_full_paths = [fp for fp in full_paths if os.path.isfile(fp)]

            if existing_full_paths:
                # Convert back to relative paths for storage in session (same style as JSON)
                existing_relative_paths = [os.path.relpath(fp, app.config['UPLOAD_FOLDER']) for fp in existing_full_paths]
                session['agent_data'][agent_id]['rag_file'] = existing_relative_paths
                session['agent_data'][agent_id]['document_name'] = [os.path.basename(fp) for fp in existing_full_paths]
            else:
                session['agent_data'][agent_id]['rag_file'] = None
                session['agent_data'][agent_id]['document_name'] = []

        elif isinstance(docs, str):
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], docs)
            if os.path.isfile(full_path):
                session['agent_data'][agent_id]['rag_file'] = [docs]  # store as list for uniformity
                session['agent_data'][agent_id]['document_name'] = [os.path.basename(docs)]
            else:
                session['agent_data'][agent_id]['rag_file'] = None
                session['agent_data'][agent_id]['document_name'] = []
        else:
            session['agent_data'][agent_id]['rag_file'] = None
            session['agent_data'][agent_id]['document_name'] = []

    return session['agent_data'][agent_id]

def agent_ask(agent_id, user_text, data):
    ts = datetime.now()
    data['history'].append({'role': 'user', 'content': user_text, 'timestamp': ts})

    messages = [
        {'role': 'system', 'content': AGENT_CONFIG[agent_id]['system_prompt']},
        *data['history'][-6:]
    ]

    # Inject RAG context only if file(s) exist
    rag_files = data.get('rag_file')
    if rag_files:
        if isinstance(rag_files, str):
            rag_files = [rag_files]
        context_parts = []
        for rel_path in rag_files:
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        context_parts.append(f.read())
                except Exception as e:
                    print(f"⚠️ Error reading {full_path}: {e}")
        if context_parts:
            context_text = "\n\n".join(context_parts)[:3000]
            messages.insert(1, {'role': 'system', 'content': f"You may use this uploaded context:\n\n{context_text}"})

    try:
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': m['role'], 'content': m['content']} for m in messages],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"⚠️ GPT API error: {e}"

    data['history'].append({'role': 'assistant', 'content': answer, 'timestamp': ts})
    session.modified = True
    return answer

def format_listing(agent_name, listing_text):
    link = chat_url_for(agent_name)
    safe_agent = escape(agent_name)
    # Keep the link on its own line so it’s visually “attached” to the listing
    return f"{listing_text}\n\n🔗 <a href='{link}'>Chat with {safe_agent}</a>"


# ─── Language Route ──────────────────────────────────────────────────────────
@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get('language')
    allowed_langs = {'en', 'af', 'de', 'ng', 'zh', 'pt'}
    if lang not in allowed_langs:
        return jsonify({'error': 'Invalid language'}), 400
    session['language'] = lang
    return jsonify({'message': 'Language set', 'language': lang})

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    lang = session.get('language', 'en')
    # Hide 'Search-AI' and 'Head of property-AI' from visible list
    visible_agents = [
        a for a in AGENT_CONFIG.keys()
        if a not in ['Search-AI', 'Head of property-AI']
    ]
    return render_template(
        'index.html',
        agent_id=None,
        agents=visible_agents,
        AGENT_CONFIG=AGENT_CONFIG,
        messages=None,
        lang=lang,
        TALLY_FORMS=TALLY_FORMS,
        datetime=datetime
    )


@app.route('/chat/<agent_id>', methods=['GET', 'POST'])
def chat(agent_id):
    if agent_id not in AGENT_CONFIG:
        flash('Unknown agent', 'error')
        return redirect(url_for('home'))

    data = user_agent_data(agent_id)
    messages = data['history']
    tally_form_url = TALLY_FORMS.get(agent_id, None)
    agent_docs = AGENT_DOCUMENTS.get(agent_id, [])

    # Handle multiple document names safely
    global_docs = load_global_docs()
    doc_list = global_docs.get(agent_id, [])
    if isinstance(doc_list, list) and len(doc_list) > 0:
        data['document_name'] = [os.path.basename(p) for p in doc_list]
    elif isinstance(doc_list, str):
        data['document_name'] = [os.path.basename(doc_list)]
    else:
        data['document_name'] = []

    # Format timestamps
    for msg in messages:
        if msg.get('timestamp') and not isinstance(msg['timestamp'], str):
            msg['timestamp'] = msg['timestamp'].strftime("%Y-%m-%d %H:%M:%S")

    lang = session.get('language', 'en')

    if request.method == 'POST':
        text = request.form.get('user_input', '').strip()
        if text:
            # Handle Search-AI behavior
            if agent_id == 'Search-AI':
                ts = datetime.now()
                data['history'].append({'role': 'user', 'content': text, 'timestamp': ts})

                # Query all visible agents (excluding Search-AI & Head of property-AI)
                internal_agents = [
                    a for a in AGENT_CONFIG.keys()
                    if a not in ['Search-AI', 'Head of property-AI']
                ]
                internal_responses = {}

                for ia in internal_agents:
                    ia_data = user_agent_data(ia)
                    agent_reply = agent_ask(ia, text, ia_data)

                    # Attach the link at the bottom of each agent’s listing
                    formatted = format_listing(ia, agent_reply)
                    internal_responses[ia] = formatted

                # Summarize and ask Head of Property-AI
                summary_prompt = [
                    "The user is searching for a property. Query:",
                    f"\"{text}\"",
                    "",
                    "Here are responses from the agents. For each listing, a 'Chat with <Agent>' link",
                    "has been appended at the bottom. IMPORTANT RULES:",
                    "1) If you shortlist or present any listing, include it verbatim with its link.",
                    "2) Do NOT remove or rewrite the links.",
                    "3) If multiple agents match, group by area/type and keep the link under each listing.",
                    ""
                ]
                for ag, resp in internal_responses.items():
                    summary_prompt.append(f"--- Agent: {ag} ---")
                    summary_prompt.append(resp)
                    summary_prompt.append("")

                head_data = user_agent_data('Head of property-AI')
                final_answer = agent_ask('Head of property-AI', "\n".join(summary_prompt), head_data)

                # Add a safety-net sources section at the bottom
                try:
                    sources_html = "<br>".join(
                        f"• <a href='{chat_url_for(a)}'>{escape(a)}</a>"
                        for a in internal_responses.keys()
                    )
                    final_answer = f"{final_answer}\n\n<hr><b>Contact the listing agents directly:</b><br>{sources_html}"
                except Exception:
                    pass

                data['history'].append({'role': 'assistant', 'content': final_answer, 'timestamp': ts})
                session.modified = True
                return redirect(url_for('chat', agent_id=agent_id))

            # Handle form trigger
            if any(phrase in text.lower() for phrase in ['leave my details', 'contact form', '📋']):
                form_link = TALLY_FORMS.get(agent_id)
                if form_link:
                    response = f"📋 Please <a href='{form_link}' target='_blank'>fill out this short form</a> so your agent can get in touch with you."
                    data['history'].append({'role': 'assistant', 'content': response, 'timestamp': datetime.now()})
                    session.modified = True
                    return redirect(url_for('chat', agent_id=agent_id))

            # Regular agent handling
            _ = agent_ask(agent_id, text, data)

        return redirect(url_for('chat', agent_id=agent_id))

    # GET method rendering
    return render_template(
        'index.html',
        agent_id=agent_id,
        messages=data['history'],
        agents=[
            a for a in AGENT_CONFIG.keys()
            if a not in ['Search-AI', 'Head of property-AI']
        ],
        AGENT_CONFIG=AGENT_CONFIG,
        TALLY_FORMS=TALLY_FORMS,
        document_name=data.get('document_name'),
        lang=lang,
        datetime=datetime,
        tally_form=tally_form_url
    )
 
@app.route('/upload/<agent_id>', methods=['GET', 'POST'])
@login_required
def upload(agent_id):
    if agent_id not in AGENT_CONFIG:
        flash('Unknown agent', 'error')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        file = request.files.get('docfile')
        if not file or file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('upload', agent_id=agent_id))
        
        if not allowed_file(file.filename):
            flash('File type not allowed', 'error')
            return redirect(url_for('upload', agent_id=agent_id))
        
        # Correct: save under /var/data/<agent_id>/
        upload_dir = os.path.join('/var/data', agent_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        # Save only the file name in JSON, not full path
        relative_path = filename
        
        global_docs = load_global_docs()
        if agent_id not in global_docs or not isinstance(global_docs[agent_id], list):
            global_docs[agent_id] = []
        global_docs[agent_id].append(relative_path)
        save_global_docs(global_docs)
        
        # Refresh in-memory docs
        load_agent_documents()
        
        flash('File uploaded', 'success')
        return redirect(url_for('chat', agent_id=agent_id))
    
    return render_template('upload.html', agent_id=agent_id)

@app.route('/reset/<agent_id>', methods=['POST'])
def reset(agent_id):
    if 'agent_data' in session and agent_id in session['agent_data']:
        session['agent_data'][agent_id]['history'] = []  # Only clears chat
        session.modified = True
    flash("Chat has been reset.", "success")
    return redirect(url_for('chat', agent_id=agent_id))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # You can also do `.strip()` to clean up input
        if (username == os.getenv('ADMIN_USERNAME') and 
            password == os.getenv('ADMIN_PASSWORD')):

            user = User(username)
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for('admin_panel'))
        else:
            flash("Invalid credentials", "error")

    return render_template('admin_login.html')

@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        abort(403)
    uploads = {}
    base = app.config['UPLOAD_FOLDER']
    if os.path.isdir(base):
        for uid in os.listdir(base):
            user_path = os.path.join(base, uid)
            if os.path.isdir(user_path):
                uploads[uid] = []
                for aid in os.listdir(user_path):
                    agent_path = os.path.join(user_path, aid)
                    if os.path.isdir(agent_path):
                        uploads[uid] += [f"{aid}/{fn}" for fn in os.listdir(agent_path)]
    return render_template('admin_panel.html', user_uploads=uploads)
@app.route('/admin/cleanup/<user_id>', methods=['POST'])
@login_required
def admin_cleanup_user(user_id):
    if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
        abort(403)  # Forbidden if not admin

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)

    if os.path.exists(user_folder):
        for filename in os.listdir(user_folder):
            file_path = os.path.join(user_folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                flash(f"Error deleting file {filename}: {str(e)}", "error")
        flash(f"All files for user {user_id} deleted successfully.", "success")
    else:
        flash(f"No files found for user {user_id}.", "info")

    return redirect(url_for('admin_panel'))

#_____Internal Queries______________________


@app.errorhandler(404)
def not_found(e):
    flash("Page not found", 'error')
    return redirect(url_for('home'))

@app.errorhandler(413)
def too_large(e):
    flash("File too large (max 16MB)", "error")
    return redirect(request.referrer or url_for('home'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    load_agent_documents()  # Load all saved agent documents into memory before starting the app
    app.run(host='0.0.0.0', port=6090)
