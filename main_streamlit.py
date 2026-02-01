"""
Mi Amigos - Digital Twin Chat Application (Streamlit Version)
A chatbot that simulates conversations with your friends using fine-tuned LoRA adapters.
"""
import os
# Fix Windows-specific PyTorch/tokenizer issues - MUST be before any torch import
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from pathlib import Path
import sys

# Get absolute paths
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
ADAPTERS_DIR = ROOT_DIR / "Adapters"

# Add directories to path for imports
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# PAGE CONFIGURATION (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Mi Amigos - Digital Twin Chat",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LAZY IMPORTS - Only import torch-related modules when needed
# ============================================================
@st.cache_resource
def get_persona_manager():
    """Lazy load persona manager to avoid torch import issues with Streamlit file watcher."""
    from persona import persona_manager
    return persona_manager


def get_available_personas():
    """Get list of available personas from Adapters folder directly."""
    personas = []
    
    # Debug info
    print(f"Looking for adapters in: {ADAPTERS_DIR}")
    
    if ADAPTERS_DIR.exists():
        for adapter_dir in ADAPTERS_DIR.iterdir():
            if adapter_dir.is_dir():
                # Skip checkpoint folders and __pycache__
                if "checkpoint" in adapter_dir.name.lower() or adapter_dir.name.startswith("_"):
                    continue
                # Check if it has adapter files
                if (adapter_dir / "adapter_model.safetensors").exists() or \
                   (adapter_dir / "adapter_model.bin").exists():
                    personas.append(adapter_dir.name)
                    print(f"  Found adapter: {adapter_dir.name}")
    else:
        print(f"  Adapters directory does not exist: {ADAPTERS_DIR}")
    
    return sorted(personas)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5em;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1em;
        margin-top: 10px;
    }
    
    /* Sidebar */
    .sidebar-header {
        padding: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 15px;
        color: #666;
        font-size: 0.9em;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_persona = None
    st.session_state.chat_history = []
    st.session_state.model_loaded = False

# ============================================================
# CORE FUNCTIONS
# ============================================================

def initialize_app():
    """Initialize the application on first load."""
    if st.session_state.initialized:
        return
    st.session_state.initialized = True


def load_persona(persona_name):
    """Load a persona/friend."""
    if not persona_name:
        return False
    
    try:
        persona_manager = get_persona_manager()
        success = persona_manager.load_persona(persona_name)
        if success:
            st.session_state.current_persona = persona_name
            st.session_state.chat_history = []
            st.session_state.model_loaded = True
            return True
        return False
    except Exception as e:
        st.error(f"❌ Error loading persona: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_response(message):
    """Generate a response from the current persona."""
    if not st.session_state.current_persona:
        return "⚠️ Please select a friend first from the sidebar!"
    
    if not message.strip():
        return ""
    
    try:
        persona_manager = get_persona_manager()
        response = persona_manager.generate_response(
            message=message,
            max_new_tokens=100,
            temperature=0.7,
        )
        return response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application function."""
    
    # Initialize app
    initialize_app()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎭 Mi Amigos</h1>
            <p>Chat with digital twins of your friends</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <h2>👤 Friend Selection</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Persona selector - get personas directly from folder
        personas = get_available_personas()
        
        if not personas:
            st.warning(f"⚠️ No adapters found!")
            st.caption(f"Looking in: `{ADAPTERS_DIR}`")
        else:
            st.success(f"Found {len(personas)} friends!")
        
        # Build options list
        options = ["-- Select a friend --"] + personas
        
        # Get current index
        current_index = 0
        if st.session_state.current_persona and st.session_state.current_persona in personas:
            current_index = personas.index(st.session_state.current_persona) + 1
        
        selected_persona = st.selectbox(
            "Choose a friend to chat with:",
            options=options,
            index=current_index,
            key="persona_selector"
        )
        
        # Load persona when changed
        if selected_persona != "-- Select a friend --":
            if selected_persona != st.session_state.current_persona:
                with st.spinner(f"🔄 Loading {selected_persona}'s model..."):
                    if load_persona(selected_persona):
                        st.success(f"✅ Now chatting with **{selected_persona}**!")
                        st.rerun()
        
        # Display current persona
        if st.session_state.current_persona:
            st.markdown(f"**Current friend:** {st.session_state.current_persona}")
        else:
            st.info("👆 Select a friend to start chatting!")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Tips
        with st.expander("💡 Tips"):
            st.markdown("""
            - Select a friend from the dropdown
            - Type messages naturally
            - The AI will respond in their style!
            """)
        
        # Info section
        st.divider()
        st.markdown(f"""
        **About:**
        Built with 💜 using:
        - Gemma 3 4B
        - LoRA adapters
        - Streamlit UI
        
        Your conversations stay 100% local and private.
        """)
    
    # ============================================================
    # MAIN CHAT AREA
    # ============================================================
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type a message...", disabled=not st.session_state.current_persona):
        if not st.session_state.current_persona:
            st.warning("⚠️ Please select a friend first from the sidebar!")
        else:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(prompt)
                st.markdown(response)
            
            # Add assistant message to chat
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update chat
            st.rerun()
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🎭 Mi Amigos - Digital Twin Chat | Powered by Gemma 3 4B + LoRA</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║            🎭 Mi Amigos - Digital Twin Chat           ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Chat with AI versions of your friends!               ║
    ║  Powered by Gemma 3 4B + Fine-tuned LoRA adapters     ║
    ║  UI: Streamlit                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    main()
