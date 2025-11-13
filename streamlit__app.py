import streamlit as st 
from pathlib import Path
from rag_system import RagSystem


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');

.main-header {
    font-size: 3rem;
    background: linear-gradient(135deg, #8B4513, #CD853F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    font-family: 'Noto Serif SC', serif;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.subtitle {
    text-align: center;
    color: #666;
    font-style: italic;
    margin-bottom: 2rem;
    font-family: 'Noto Serif SC', serif;
}

.chat-message {
    padding: 1.2rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    animation: fadeIn 0.5s ease-in;
}

.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 4px solid #2196f3;
    margin-left: 2rem;
}

.assistant-message {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 4px solid #8B4513;
    margin-right: 2rem;
}

.source-info {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    padding: 0.8rem;
    border-radius: 0.8rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    border: 1px solid #ce93d8;
    transition: all 0.3s ease;
}

.source-info:hover {
    background: linear-gradient(135deg, #e8eaf6, #c5cae9);
    border-color: #9c27b0;
}

.status-success {
    color: #2e7d32;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.status-error {
    color: #d32f2f;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.status-warning {
    color: #f57c00;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.example-button {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border: 1px solid #ffb74d;
    border-radius: 0.5rem;
    padding: 0.5rem;
    margin: 0.2rem 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.example-button:hover {
    background: linear-gradient(135deg, #ffe0b2, #ffcc80);
    transform: translateX(5px);
}

.sidebar-section {
    background: linear-gradient(135deg, #fafafa, #f5f5f5);
    padding: 1rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    border: 1px solid #e0e0e0;
}

.typing-indicator {
    display: inline-block;
    animation: typing 1.5s infinite;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes typing {
    0%, 60%, 100% { opacity: 1; }
    30% { opacity: 0.5; }
}

.metric-card {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    padding: 1rem;
    border-radius: 0.8rem;
    text-align: center;
    margin: 0.5rem 0;
    border: 1px solid #81c784;
}

.progress-bar {
    background: linear-gradient(90deg, #4caf50, #8bc34a);
    height: 4px;
    border-radius: 2px;
    animation: progress 2s ease-in-out;
}

@keyframes progress {
    from { width: 0%; }
    to { width: 100%; }
}
</style>
""", unsafe_allow_html=True)

def display_chat_message(role, content, sources=None, typing=False):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋‍♀️ 您:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        typing_indicator = '<span class="typing-indicator">💭</span>' if typing else '🤖'
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>{typing_indicator} 红楼梦助手:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

        if sources:
            with st.expander(f"📖 参考文档片段 ({len(sources)}个)", expanded=False):
                for i, source in enumerate(sources, 1):
                    similarity_color = "#4caf50" if source['similarity'] > 0.5 else "#ff9800" if source[
                                                                                                     'similarity'] > 0.3 else "#f44336"
                    st.markdown(f"""
                    <div class="source-info">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>📄 片段 {i}: {source['source']}</strong>
                            <span style="background: {similarity_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 1rem; font-size: 0.8rem;">
                                相似度: {source['similarity']:.3f}
                            </span>
                        </div>
                        <div style="background: rgba(255,255,255,0.8); padding: 0.5rem; border-radius: 0.5rem; border-left: 3px solid {similarity_color};">
                            <em>📝 内容预览:</em><br>
                            {source['content_preview']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def getEnvinfo(key):
    env_file = Path(__file__).parent / ".env"

    env_lines = []
    env_res=None
    
    if env_file.exists():
        with open(env_file, "r") as f:
            env_lines = f.readlines()

    for i, line in enumerate(env_lines):
        if line.strip().startswith(f"{key}="):
            env_res = line.strip().split("=")[1]

    return env_res


def check_cache_exists():
    cache_dir=Path("cache")
    vector_cache=cache_dir/"vector_cache.pkl"
    chunk_cache=cache_dir/"chunk_cache.pkl"
    return vector_cache.exists() and chunk_cache.exists()


def init_session_state():
    #系统状态
    if 'system_status' not in st.session_state:
        st.session_state.system_status = False
    #API密钥
    if 'api_key' not in st.session_state:
        st.session_state.api_key = getEnvinfo('DEEPSEEK_API_KEY')
    #预设问题
    if 'preset_question' not in st.session_state:
        st.session_state.preset_question = ""
    #选择角色
    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = ""

    #RAG系统
    rag=RagSystem(st.session_state.api_key)
    rag.initialize()
    st.session_state.rag_system=rag
    st.session_state.system_status=True
 
def main():
    # st.header("论文检索助手")
    init_session_state()
    st.markdown('<h1 class="main-header">论文检索助手</h1>', unsafe_allow_html=True)

    # print(getEnvinfo('DEEPSEEK_API_KEY'),'DEEPSEEK_API_KEY')


    api_key = getEnvinfo('DEEPSEEK_API_KEY')

    #侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        api_key_input=st.text_input("🔑 DeepSeek API密钥",
         type="password",
         value=st.session_state.api_key,
         help="请输入AI API_KEY,可在硅基流动平台（https://cloud.siliconflow.cn/i/PyAFBgHG）申请，点击链接注册即可免费领token",
         placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ")

        if api_key_input:
            st.session_state.api_key=api_key_input
        else:
            st.session_state.system_status=False

        st.header("📊 系统状态")

        if not st.session_state.system_status:
            st.markdown('<p class="status-warning"> ⚠️系统未初始化</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-success"> ✅系统已就绪</p> ', unsafe_allow_html=True)
         
            if st.session_state.rag_system:
                col1,col2=st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                        <h3>📄</h3>
                        <p>{len(st.session_state.rag_system.documents)}</p>
                        <small>文档数量</small>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                        <h3>📝</h3>
                        <p>{len(st.session_state.rag_system.document_chunks)}</p>
                        <small>文档块</small>
                        </div>
                    """, unsafe_allow_html=True)

        #获取状态按钮
        cache_exists=check_cache_exists()
        if cache_exists and st.session_state.system_status:
            st.button("🔄 重新初始化系统",use_container_width=True)
        else:
            if st.button("🚀 加载缓存数据",use_container_width=True):
                init_session_state()

        #搜索参数配置
        if st.session_state.system_status:
            st.header("🔍 搜索参数设置")
            #最大返回文档数量
            top_k=st.slider(
                '📄 最大返回文档块数量',
                min_value=1,
                max_value=20,
                value=10,
                help='设置搜索时返回多少个相关文档块'
            )

            #相似度值域
            similarity_threshold=st.slider(
                ' 🔍 相似度值域',
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.01,
                help='只返回相似度大于该值的文档块，值越高越精准'
            )

            st.session_state.search_top_k=top_k
            st.session_state.search_similarity_threshold=similarity_threshold

    #对话区域
    st.header("💬 对话区域")

    #快捷问题
    st.subheader("💡 快捷问题")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        if st.button("🎭人物关系",use_container_width=True):
            if st.session_state.system_status:
                #设置预设问题到session_state
                st.session_state.preset_question="红楼梦人物之间的关系是什么？"
                st.session_state.selected_role='人物关系'
    with col2:
        if st.button("🎨情节概要",use_container_width=True):
            if st.session_state.system_status:
                #设置预设问题到session_state
                st.session_state.preset_question="红楼梦的情节概要是什么？"
                st.session_state.selected_role='情节概要'
    with col3:
        if st.button("💎文学手法  ",use_container_width=True):
            if st.session_state.system_status:
                #设置预设问题到session_state
                st.session_state.preset_question="红楼梦的文学手法有哪些？"
                st.session_state.selected_role='文学手法'
    with col4:
        if st.button("🧹文学价值",use_container_width=True):
            if st.session_state.system_status:
                #设置预设问题到session_state
                st.session_state.preset_question="红楼梦的文学价值是什么？"
                st.session_state.selected_role='文学价值'
    
    #显示当前选择角色
    if st.session_state.selected_role:
        st.info(f"🎯 当前选择角色: {st.session_state.selected_role}")

    with st.form(key='chat_form',clear_on_submit=True):
        col1,col2=st.columns([5,1])
        with col1:
            user_input=st.text_area(
                "💬 输入你的问题",
                value=st.session_state.preset_question,
                height=100,
                max_chars=500,
                placeholder="请输入你的问题",
                help="输入你的问题，点击发送按钮或按回车键发送",
                label_visibility="visible"
            )
        with col2:
            st.markdown("<br/>",unsafe_allow_html=True)
            submit_button=st.form_submit_button("🚀 发送",use_container_width=True,type="primary")
            clear_button=st.form_submit_button("🧹 清除",use_container_width=True)

        if clear_button:
            st.session_state.preset_question=""
            st.session_state.selected_role=""

        #处理发送按钮
        if submit_button and user_input.strip():
            #如果系统状态和RAG系统都正常，则发送问题
            if st.session_state.system_status and st.session_state.rag_system:
                #先清除预设问题和选择角色
                st.session_state.preset_question=""
                st.session_state.selected_role=""

                #显示对话效果
                display_chat_message("user", user_input, typing=True)

                #显示思考状态
                thinking_placeholder=st.empty()
                with thinking_placeholder:
                    display_chat_message("assistant", "正在思考...", typing=True)

                with st.spinner("正在搜索相关文档..."):
                    #获取搜索参数
                    top_k=getattr(st.session_state, 'search_top_k', 10)
                    similarity_threshold=getattr(st.session_state, 'search_similarity_threshold', 0.01)
                    result=st.session_state.rag_system.query(user_input, top_k=top_k, similarity_threshold=similarity_threshold)

                #清除思考状态
                thinking_placeholder.empty()

                processed_sources=[]
                for source in result['sources']:
                    if 'content_preview' in source:
                        content=source['content_preview']
                    elif 'content' in source:
                        content=source['content'][:300]+'...' if len(source['content'])>300 else source['content']
                    else:
                        content='无内容预览'

                    processed_sources.append({
                        'source':source['source'],
                        'similarity':source['similarity'],
                        'content_preview':content
                    })

                display_chat_message("assistant", result['answer'], processed_sources, typing=False)

if __name__ == "__main__":
    main()