import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableBranch
from dotenv import load_dotenv

from preprocess import extract_audio, extract_frames, detect_pauses
from voice_agent import run_voice_agent
from face_agent import run_face_agent
from text_agent import run_text_agent
from cidf_agent import run_cidf_agent
from coach_agent import run_coach_agent

load_dotenv()

memory = InMemoryChatMessageHistory()

def run_preprocessing(_):
    print("\n--- STEP 1: PREPROCESSING ---")
    extract_audio()
    extract_frames()
    pauses = detect_pauses()
    return {"pauses": pauses, "status": "preprocessing_complete"}

def run_all_agents(_):
    print("\n--- STEP 2: RUNNING ALL AGENTS ---")
    print("Running Voice Agent...")
    voice_results = run_voice_agent()
    print("Running Face Agent...")
    face_results = run_face_agent()
    print("Running Text Agent...")
    text_results = run_text_agent()
    return {
        "voice_results": voice_results,
        "face_results": face_results,
        "text_results": text_results,
        "status": "agents_complete"
    }

def run_cidf(_):
    print("\n--- STEP 3: RUNNING CIDF ---")
    cidf_results = run_cidf_agent()
    return cidf_results

def should_coach(cidf_results):
    return cidf_results.get("incongruence_score", 0) >= 0.5

coaching_branch = RunnableBranch(
    (
        RunnableLambda(should_coach),
        RunnableLambda(lambda x: {
            **x,
            "coaching": run_coach_agent(),
            "coaching_triggered": True
        })
    ),
    RunnableLambda(lambda x: {
        **x,
        "coaching": "No coaching needed - communication is congruent",
        "coaching_triggered": False
    })
)

def save_session_to_memory(results):
    memory.add_user_message("TrueSignal analysis complete")
    memory.add_ai_message(
        f"Incongruence score: {results.get('incongruence_score')}. "
        f"Conflict: {results.get('conflict_description')}. "
        f"Coaching triggered: {results.get('coaching_triggered')}"
    )
    return results

def run_pipeline():
    print("=" * 50)
    print("TRUESIGNAL PIPELINE STARTING")
    print("=" * 50)

    preprocessing_step = RunnableLambda(run_preprocessing)
    agents_step = RunnableLambda(run_all_agents)
    cidf_step = RunnableLambda(run_cidf)
    memory_step = RunnableLambda(save_session_to_memory)

    pipeline = (
        preprocessing_step
        | agents_step
        | cidf_step
        | coaching_branch
        | memory_step
    )

    final_results = pipeline.invoke({})

    print("\n" + "=" * 50)
    print("TRUESIGNAL PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Incongruence Score: {final_results.get('incongruence_score')}")
    print(f"Dominant Conflict: {final_results.get('conflict_description')}")
    print(f"Coaching Triggered: {final_results.get('coaching_triggered')}")
    if final_results.get('coaching_triggered'):
        print(f"Coaching: {final_results.get('coaching')}")

    return final_results

if __name__ == "__main__":
    run_pipeline()
