from flask import Flask, render_template, request, session, send_from_directory
from services.rag_service import get_rag_response
import re
import os

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

@app.route("/")
def home():

    session["messages"] = []

    return render_template("index.html")

@app.route("/documents/<filename>")
def serve_document(filename):
    return send_from_directory("documents", filename)


@app.route("/chat", methods=["POST"])
def chat():
    print("\n========== CHAT ROUTE ENTERED ==========", flush=True)

    prompt = request.form.get("prompt", "").strip()

    if not prompt:
        return render_template("index.html", messages=session.get("messages", []))

    # Get existing conversation
    messages = session.get("messages", [])

    response, sources, show_sources = get_rag_response(prompt, messages)

    # Remove duplicate source documents
    unique_sources = []
    seen_sources = set()

    for source in sources:
        source_name = source.get("source")

        if source_name and source_name not in seen_sources:
            unique_sources.append(source)
            seen_sources.add(source_name)

    sources = unique_sources

    # =========================================================
    # PARSE STRUCTURED LLM RESPONSE
    # =========================================================

    print("\n================ LLM RESPONSE ================\n")
    print(response)
    print("\n==============================================\n")

    # Response Type
    response_type_match = re.search(
        r"^\s*(INFORMATIONAL|COMPLIANT|NON-COMPLIANT|NEEDS REVIEW|OUT OF SCOPE)\s*$",
        response,
        re.IGNORECASE | re.MULTILINE,
    )

    if response_type_match:
        response_type = response_type_match.group(1).upper()
    else:
        # If there is no structured response type,
        # treat it as a conversational informational answer.
        response_type = "INFORMATIONAL"

    # Answer
    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?=\nWHY:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    answer = answer_match.group(1).strip() if answer_match else ""


    # Why
    why_match = re.search(
        r"WHY:\s*(.*?)(?=\nWHAT YOU CAN DO:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    why = why_match.group(1).strip() if why_match else ""


    # What You Can Do
    action_match = re.search(
        r"WHAT YOU CAN DO:\s*(.*?)(?=\nPOLICY CONTEXT:|\nUSER SITUATION:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    what_you_can_do = action_match.group(1).strip() if action_match else ""


    # Summary
    summary_match = re.search(
        r"SUMMARY:\s*(.*?)(?=\nREASON:|\Z)", response, re.IGNORECASE | re.DOTALL
        )

    summary = summary_match.group(1).strip() if summary_match else ""


    # Reason
    reason_match = re.search(
        r"REASON:\s*(.*?)(?=\nRECOMMENDATION:|\nPOLICY REQUIREMENT:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    reason = reason_match.group(1).strip() if reason_match else ""


    # Recommendation
    recommendation_match = re.search(
        r"RECOMMENDATION:\s*(.*?)(?=\nPOLICY REQUIREMENT:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    recommendation = recommendation_match.group(1).strip() if recommendation_match else ""


    # Policy Requirement
    requirement_match = re.search(
        r"POLICY REQUIREMENT:\s*(.*?)(?=\nPOLICY CONTEXT:|\nUSER SITUATION:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    policy_requirement = requirement_match.group(1).strip() if requirement_match else ""


    # Add current question and answer
    messages.append({"role": "user", "content": prompt})

    messages.append ({
    'role': 'assistant',
    'content': response,
    'compliance': response_type,
    'answer': answer,
    'why': why,
    'what_you_can_do': what_you_can_do,
    'summary': summary,
    'reason': reason,
    'policy_requirement': policy_requirement,
    'recommendation': recommendation,
    'sources': sources,
    "show_sources": show_sources
    })

    print("\n========== MESSAGE HISTORY ==========")

    for i, msg in enumerate(messages):
        print(i, msg.get("role"), msg.get("content", "")[:100])

    print("MESSAGE COUNT:", len(messages))
    print("SESSION SIZE APPROX:", len(str(messages)))

    print("=====================================\n")
        

    session["messages"] = messages
    session.modified = True

    print("\n========== MESSAGE HISTORY ==========")

    for i, msg in enumerate(messages):
        print(i, msg.get("role"), msg.get("content", "")[:100])

    print("=====================================\n")

    # Render the SAME UI with the conversation
    return render_template("index.html", messages=messages)


if __name__ == "__main__":
    app.run()
