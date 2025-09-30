# ADRDS — Audience-Driven Real-Time Data Storytelling

A **Streamlit** prototype for audience-driven, real-time data storytelling.  
The app implements a lightweight version of the **ADRDS framework** described in the paper *Towards Audience-Driven Real-Time Data Storytelling*, featuring a **decision tree flow** and synchronized **presenter/audience** modes.

---

## ✨ Features

- **Presenter & Audience modes** in the same Streamlit app (toggle in sidebar).
- **Room codes** to separate sessions.
- **Decision tree** story structure loaded from `story.json`.
- **Live voting** from the audience.
- **Real-time metrics** to support branching decisions:
  - *Coverage* (participation rate),
  - *Consensus* `p_max` (largest option share),
  - *Confidence interval* 95% (bootstrap).
- **Visual summaries** (Altair bar charts).
- **Human override**: presenter can force progression even if thresholds aren’t met.

---

## 🗂 Architecture (at a glance)

- **PSI — Participatory Sensing Interface**: audience view for voting. :contentReference[oaicite:1]{index=1}  
- **PCI — Presenter Control Interface**: presenter dashboard showing current node, votes, metrics, and charts. :contentReference[oaicite:2]{index=2}  
- **DCE — Decision Control Engine**: logic that evaluates *coverage*, *consensus*, and *CI95%*. :contentReference[oaicite:3]{index=3}  
- **SKB — Story Knowledge Base**: stories defined in `story.json`, typically modeled with a **three-act structure** (setup, conflict, resolution). :contentReference[oaicite:4]{index=4}  

> Demo note: the app uses **in-memory storage** only. For real-world use, connect to Redis/DB for persistence.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ recommended
- Up-to-date `pip`

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt```

### Run the app
```python3 -m streamlit run app.py```

## 🧭 How to Use

1. **Open the app** and select **Presenter mode** in the sidebar.  
   - A **room code** is generated (or entered manually).  
   - The presenter sees the **current node**, vote distribution, and metrics.

2. **Share the room code** with the audience.  
   - Audience members join the same app, switch to **Audience mode**, and enter the room code.

3. **Voting**  
   - Audience members cast their votes.  
   - The presenter’s view updates in real time.

4. **Branching**  
   - Once thresholds are satisfied (or manually overridden), the presenter advances to the next node in the **decision tree** (`story.json`).

---

## 📄 Story Definition (`story.json`)

**Minimal example:**

```json
{
  "start": "setup",
  "nodes": {
    "setup": {
      "id": "setup",
      "text": "When you hear 'objects launched into space', what interests you most?",
      "options": [
        { "id": "technical", "text": "Precise data and growth models" },
        { "id": "general",   "text": "Stories and impacts on daily life" }
      ],
      "next_map": {
        "technical": "tech_act1",
        "general": "gen_act1"
      }
    }
  }
}

