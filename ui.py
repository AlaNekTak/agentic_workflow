import gradio as gr
from app import build_agent 

agent = build_agent()


# ─── GRADIO UI ───────────────────────────────────────────────────────
def dispatch(msg, hist, trace, thumbs_state):
    """Handle a user submit/click event."""
    # ── Blank submit → just echo current state ─────────────────────
    if not msg.strip():
        return gr.update(), hist, trace, gr.update(visible=False)
    
    chat_history = hist.copy() 
    chat_history.append({"role": "user", "content": msg})
    
    # ── Run the agent ──────────────────────────────────────────────
    resp   = agent.invoke({"messages": chat_history})
    answer      : str           = resp["output"]["answer"]
    new_thumbs  : list[tuple]    = resp["output"]["thumbs"]
    new_traces  : list[str]      = resp["output"]["traces"]

    chat_history.append({"role": "assistant", "content": answer})
    hist = chat_history   

    # ── Gallery update (show only if thumbs exist) ─────────────────
    gallery_update = (
        gr.update(value=new_thumbs, visible=True)
        if new_thumbs else gr.update(visible=False)
    )

    # ── Trace textbox update ───────────────────────────────────────
    if new_traces:
        trace += "\n".join(new_traces) + "\n"
    else:
        trace += "\nChat step:\n"+ f"User: {msg}\n" + f"Assistant: {answer}\n"

    # clear the input box, return updated components
    return "", hist, trace, gallery_update

demo_q = ["Which BRCA1 frameshift variants are linked to hereditary breast cancer?",
          "How many papers mention cancer in their title?",
          "What is the genetic evidence for the involvement of TP53 (p53) in colorectal cancer progression?",
          "Which specific mutations in BRCA1 have been strongly linked to hereditary breast cancer, and what mechanisms underlie their pathogenicity?",
          "Which CFTR variants have the strongest correlation with cystic fibrosis severity, and how do they affect chloride channel function?"
           ]

trace = gr.Textbox(
    value="",                     # start blank
    label="Backend trace",
    lines=24,                     # visible height
    autoscroll=True,              # scrollbar always visible
    interactive=False,
)

HEADER_CSS = """
.header    { align-items: center; }
.header h1 { font-size: 1.8rem !important; }   /* bigger title */
.header h2 { font-size: 1.3rem !important; }   /* bigger subtitle if any */
"""

with gr.Blocks(css=HEADER_CSS) as front:
    with gr.Row(elem_classes="header"):
        gr.Markdown("# Research-Agent Demo")
        gr.Image("pic.png", width=64, height=64, show_download_button=False, container=False)

    question_box = gr.Dropdown(
        label="Question",
        choices=demo_q,
        value=demo_q[0],
        interactive=True,
        scale=5
    )
    chatbot = gr.Chatbot(type="messages", label="Assistant", render_markdown=True, sanitize_html=False)
    gallery = gr.Gallery(
    label="PDF Previews",
    columns=4,
    height="auto",
    show_download_button=False,
    visible=False
    )
    user_in = gr.Textbox(lines=1, placeholder="Ask something…")
    
    send    = gr.Button("Send")
    restart = gr.ClearButton(components=[user_in, chatbot, trace, gallery])

    send.click(dispatch, [user_in, chatbot, trace, gallery], [user_in, chatbot, trace, gallery])
    user_in.submit(dispatch, [user_in, chatbot, trace, gallery], [user_in, chatbot, trace, gallery])

with gr.Blocks(css=HEADER_CSS) as back:
        with gr.Accordion("Flow tracing", open=False):
            trace.render()

demo = gr.TabbedInterface(
    [front, back],
    ["Interface", "Backend"]
)

demo.launch(debug=True)