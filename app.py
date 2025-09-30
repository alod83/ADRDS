# ADRDS Streamlit Prototype
# -----------------------------------------------------------
# Audience-Driven Real-Time Data Storytelling (ADRDS)
# Streamlit prototype based on the attached paper.
# Run with:  streamlit run app.py
# -----------------------------------------------------------

import os
import time
import random
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import io

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependency: qrcode (for QR code in presenter view)
try:
    import qrcode
except Exception:
    qrcode = None

APP_TITLE = "ADRDS — Audience-Driven Real-Time Data Storytelling"

# ---------------------------- Utilities ---------------------------- #

def get_store() -> Dict:
    """Global in-memory store shared across sessions.
    WARNING: This is a simple in-memory store for demo purposes.
    In production, replace with a database or persistent cache.
    """
    @st.cache_resource(show_spinner=False)
    def _store():
        return {}
    return _store()


def make_room_code(n: int = 5) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(n))


def get_user_id() -> str:
    if "uid" not in st.session_state:
        st.session_state["uid"] = str(uuid.uuid4())
    return st.session_state["uid"]


def hash_user_id(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:16]


# ---------------------------- Data Model ---------------------------- #

@dataclass
class Thresholds:
    coverage: float = 0.5      # τc minimum proportion of audience that must respond
    consensus: float = 0.55    # τcons minimum share for top option
    ci_width: float = 0.50     # τci maximum width of 95% bootstrap CI for p_max

@dataclass
class Option:
    key: str
    label: str

@dataclass
class Node:
    id: str
    text: str
    options: List[Option]
    next_map: Dict[str, str]  # maps option.key -> next node id
    thresholds: Thresholds = field(default_factory=Thresholds)
    terminal: bool = False
    ctas: Optional[List[str]] = None  # only for terminal nodes

@dataclass
class DecisionTree:
    nodes: Dict[str, Node]
    start: str

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]


# ---------------------------- Demo Story ---------------------------- #

import json
from typing import Dict, Any

def demo_tree() -> DecisionTree:
    """Load a decision tree from story.json file."""
    
    # Carica il file JSON
    try:
        with open('story.json', 'r', encoding='utf-8') as file:
            tree_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Il file story.json non è stato trovato")
    except json.JSONDecodeError:
        raise ValueError("Il file story.json contiene JSON non valido")
    
    # Costruisci i nodi dal JSON
    nodes = {}
    
    for node_id, node_data in tree_data["nodes"].items():
        # Crea le opzioni
        options = []
        for option_data in node_data["options"]:
            options.append(Option(option_data["id"], option_data["text"]))
        
        # Crea il nodo
        node = Node(
            id=node_data["id"],
            text=node_data["text"],
            options=options,
            next_map=node_data["next_map"],
            terminal=node_data.get("terminal", False),
            ctas=node_data.get("ctas", [])
        )
        
        nodes[node_id] = node
    
    return DecisionTree(nodes=nodes, start=tree_data["start"])


# Funzione alternativa più generica per caricare da qualsiasi file
def load_tree_from_file(filename: str) -> DecisionTree:
    """Load a decision tree from a JSON file."""
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            tree_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Il file {filename} non è stato trovato")
    except json.JSONDecodeError:
        raise ValueError(f"Il file {filename} contiene JSON non valido")
    
    nodes = {}
    
    for node_id, node_data in tree_data["nodes"].items():
        options = []
        for option_data in node_data["options"]:
            options.append(Option(option_data["id"], option_data["text"]))
        
        node = Node(
            id=node_data["id"],
            text=node_data["text"],
            options=options,
            next_map=node_data["next_map"],
            terminal=node_data.get("terminal", False),
            ctas=node_data.get("ctas", [])
        )
        
        nodes[node_id] = node
    
    return DecisionTree(nodes=nodes, start=tree_data["start"])


# ---------------------------- Metrics & Logic ---------------------------- #

def compute_distribution(votes: List[str], options: List[Option]) -> Tuple[pd.DataFrame, float, float, Tuple[float, float]]:
    """Return (df, coverage, p_max, (ci_lo, ci_hi))."""
    if not votes:
        df = pd.DataFrame({"option": [o.label for o in options], "count": [0]*len(options), "pct": [0.0]*len(options)})
        return df, 0.0, 0.0, (0.0, 1.0)

    counts = {o.key: 0 for o in options}
    for v in votes:
        if v in counts:
            counts[v] += 1
    total = max(1, sum(counts.values()))

    rows = []
    for o in options:
        c = counts[o.key]
        rows.append({"key": o.key, "option": o.label, "count": c, "pct": c/total})
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    p_max = float(df.iloc[0]["pct"]) if not df.empty else 0.0

    # Bootstrap CI for p_max (95%)
    boot = []
    votes_idx = [o.key for o in options]
    encoded = [votes_idx.index(v) for v in votes if v in votes_idx]
    if len(encoded) > 0:
        for _ in range(min(1000, 200 + 20*len(votes))):
            sample = np.random.choice(encoded, size=len(encoded), replace=True)
            # recompute p_max on bootstrap sample
            b_counts = np.bincount(sample, minlength=len(options))
            b_total = max(1, b_counts.sum())
            boot.append(b_counts.max()/b_total)
        lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    else:
        lo, hi = 0.0, 1.0

    return df, 1.0, p_max, (lo, hi)


def ready_to_branch(n_participants: int, votes: List[str], node: Node, audience_size: int) -> bool:
    if audience_size <= 0:
        return False
    coverage = len(votes) / audience_size
    df, _, p_max, (ci_lo, ci_hi) = compute_distribution(votes, node.options)
    ci_width = ci_hi - ci_lo
    return (coverage >= node.thresholds.coverage) and (p_max >= node.thresholds.consensus) and (ci_width <= node.thresholds.ci_width)


def select_next_node(node: Node, votes: List[str]) -> Optional[str]:
    if not votes:
        return None
    counts = {k: 0 for k in node.next_map.keys()}
    for v in votes:
        if v in counts:
            counts[v] += 1
    winner_key = max(counts.items(), key=lambda x: x[1])[0]
    return node.next_map.get(winner_key)


# ---------------------------- CTA Selection ---------------------------- #

def cta_utility(cta: str, path: List[str], readiness_share: float) -> float:
    # Very simple heuristic demo
    fit = 1.0 if "conflict_boundaries" in path or "resolution_change" in path else 0.7
    readiness = readiness_share  # proportion who said "yes"
    risk = 0.2 if "notifiche" in cta.lower() else 0.1
    alpha, beta, gamma = 0.6, 0.5, 0.4
    return alpha*fit + beta*readiness - gamma*risk


def choose_cta(ctas: List[str], path: List[str], votes: List[str], yes_key: str = "yes") -> str:
    if not ctas:
        return ""
    if votes:
        readiness_share = votes.count(yes_key)/len(votes)
    else:
        readiness_share = 0.0
    scored = [(cta, cta_utility(cta, path, readiness_share)) for cta in ctas]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ---------------------------- Session State ---------------------------- #

def get_or_create_session(room: str) -> Dict:
    store = get_store()
    if room not in store:
        store[room] = {
            "created": time.time(),
            "tree": demo_tree(),
            "current": "setup",
            "path": ["setup"],
            "votes": {},            # node_id -> { user_hash: option_key }
            "log": [],
        }
    return store[room]


def submit_vote(session: Dict, node_id: str, uid_hash: str, option_key: str):
    session.setdefault("votes", {}).setdefault(node_id, {})[uid_hash] = option_key


# ---------------------------- UI Blocks ---------------------------- #

def header():
    st.set_page_config(page_title=APP_TITLE, page_icon="📊")
    #st.title(APP_TITLE)
    #st.caption("A Prototype of an Interactive Presentation Guided by the Audience")


def presenter_view(room: str, session: Dict):
    st.title(":blue[Presenter Control Interface]")
    #st.subheader("Presenter Modality")
    join_url = f"?room={room}&role=audience"

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown(f"**Room Code:** `{room}`")
        st.markdown("Link for the audience:")
        st.code(join_url, language="text")
        st.markdown(f"[Open in a new tab]({join_url})")
        
    tree: DecisionTree = session["tree"]
    node = tree.get(session["current"])

    with col2:
        st.markdown("**Current Question**")
        st.write(node.text)
        st.markdown("**Number of partipants who have already answered:** ")
        participants_current = len(session.get("votes", {}).get(node.id, {}))
        st.metric(label="Users (answerers)", value=participants_current)

    # Results panel
    votes = list(session.get("votes", {}).get(node.id, {}).values())
    df, coverage, p_max, (ci_lo, ci_hi) = compute_distribution(votes, node.options)

    st.markdown("### :blue[Real-time results]")
    if df is not None and len(df) > 0:
        try:
            import altair as alt
            base = alt.Chart(df).mark_bar(color='#0253A3').encode(
                x=alt.X("option:N", title="", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Votes"),
                tooltip=["option", "count"]
            )
            st.altair_chart(base, use_container_width=True)
        except Exception:
            st.bar_chart(df.set_index("option")["count"])
    else:
        st.info("No votes yet.")

    st.markdown(
        f"**Responders**: {participants_current} — **Coverage (responders)**: {coverage:.2f} — "
        f"**Consensus p_max**: {p_max:.2f} — **CI95%**: [{ci_lo:.2f}, {ci_hi:.2f}]"
    )

    can_auto = ready_to_branch(
        participants_current,
        votes,
        node,
        max(1, participants_current)
    )

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Go Ahead (force) →"):
            nxt = select_next_node(node, votes) or list(node.next_map.values())[0]
            session["current"] = nxt
            session["path"].append(nxt)
            st.rerun()
    with c2:
        if st.button("Reset node"):
            session.get("votes", {}).pop(node.id, None)
            st.rerun()
    with c3:
        if st.button("Go ahead if thresholds are OK"):
            if can_auto:
                nxt = select_next_node(node, votes)
                if nxt:
                    session["current"] = nxt
                    session["path"].append(nxt)
                    st.rerun()
            else:
                st.warning("Thresholds not satisfied.")

    if node.terminal:
        st.success("Reached terminal node")
        if node.ctas:
            res_votes = list(session.get("votes", {}).get("resolution_change", {}).values())
            best_cta = choose_cta(node.ctas, session["path"], res_votes)
            st.markdown("#### Suggested CTA")
            st.write(best_cta)

    with st.expander("Structure and Path"):
        st.json({
            "current": session["current"],
            "path": session["path"],
            "votes": {k: {"total": len(v), "options": v} for k, v in session.get("votes", {}).items()},
        })


def audience_view(room: str, session: Dict):
    st.title(":orange[Participatory Sensing Interface]")
    #st.subheader("Audience Modality")
    st.markdown(f"**Room:** `{room}`")

    uid = get_user_id()
    uid_hash = hash_user_id(uid)

    tree: DecisionTree = session["tree"]
    node = tree.get(session["current"])

    st.markdown("### :orange[Current Question]")
    st.write(node.text)

    previous = session.get("votes", {}).get(node.id, {}).get(uid_hash)
    if previous:
        st.info("You have already voted. You can change your answer")

    chosen = st.radio(
        "Select an option:",
        options=[o.key for o in node.options],
        format_func=lambda k: next(o.label for o in node.options if o.key == k),
        index=[o.key for o in node.options].index(previous) if previous else 0,
    )
    if st.button("Send/Update your vote"):
        submit_vote(session, node.id, uid_hash, chosen)
        st.success("Vote Saved!")
        st.rerun()

    votes = list(session.get("votes", {}).get(node.id, {}).values())
    df, _, _, _ = compute_distribution(votes, node.options)
    if not df.empty:
        st.markdown("#### Current Distribution (anonymous)")
        st.progress(min(0.99, float(df.iloc[0]['pct'])))


# ---------------------------- Main App ---------------------------- #

def main():
    header()

    # URL parameters
    qp = st.query_params
    role = (qp.get("role", "presenter") or "presenter").lower()
    room = qp.get("room", make_room_code())

    session = get_or_create_session(room)

    # Sidebar controls
    st.sidebar.header("Session Settings")
    st.sidebar.text_input("Room code", value=room, key="room_input")
    new_room = st.sidebar.button("Create a new room")
    if new_room:
        st.query_params = {"room": make_room_code(), "role": "presenter"}
        st.rerun()

    mode = st.sidebar.radio("Ruolo", ["presenter", "audience"], index=0 if role == "presenter" else 1)
    if mode != role:
        st.query_params = {"room": room, "role": mode}
        st.rerun()

    if mode == "presenter":
        presenter_view(room, session)
    else:
        audience_view(room, session)


if __name__ == "__main__":
    main()
