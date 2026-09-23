from langchain_chroma import Chroma
import json
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# from config import GOOGLE_API_KEY
from config import embeddings, POLICY_METADATA
from services.openai_service import get_response

# Load the Chroma database
db = Chroma(persist_directory="vectorstore", embedding_function=embeddings)

# Load chunks saved during ingestion
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

documents = [
    Document(page_content=item["page_content"], metadata=item["metadata"])
    for item in chunks_data
]

def generate_self_query(user_query):
    """
    Uses the LLM to convert the user's natural-language question
    into:

    1. A better search query
    2. Metadata filters for retrieval
    """

    # Get metadata values dynamically from config.py
    departments = sorted(
        {
            metadata.get("department")
            for metadata in POLICY_METADATA.values()
            if metadata.get("department")
        }
    )

    policy_types = sorted(
        {
            metadata.get("policy_type")
            for metadata in POLICY_METADATA.values()
            if metadata.get("policy_type")
        }
    )

    versions = sorted(
        {
            metadata.get("version")
            for metadata in POLICY_METADATA.values()
            if metadata.get("version")
        }
    )

    effective_dates = sorted(
        {
            metadata.get("effective_date")
            for metadata in POLICY_METADATA.values()
            if metadata.get("effective_date")
        }
    )

    self_query_prompt = f"""
    You are a query analyzer for PolicyLens AI, an internal company policy retrieval system.

    Your job is NOT to answer the user's question.

    Your job is to prepare the question for policy retrieval.

    USER QUESTION:
    {user_query}

    AVAILABLE METADATA VALUES:

    Departments:
    {departments}

    Policy Types:
    {policy_types}

    Versions:
    {versions}

    Effective Dates:
    {effective_dates}

    Allowed metadata fields:
    - department
    - policy_type
    - version
    - effective_date

    RULES:

    1. PRESERVE THE USER'S SEARCH INTENT.

    The USER QUESTION may already have been rewritten using
    conversation history.

    Do NOT broaden, generalize, or change its meaning.

    Do NOT replace a specific question with a broad topic search.

    2. The "search_query" should remain as close as possible
    to the USER QUESTION.

    You may make only small changes needed to make the query
    clear for document retrieval.

    3. Do NOT add new topics, responsibilities, roles, approvals,
    requirements, or concepts that are not present in the
    USER QUESTION.

    4. Do NOT turn a specific question into a general policy search.

    Example:

    USER QUESTION:
    "How many sick and wellness leave days do managers get per year?"

    Good search_query:
    "How many sick and wellness leave days do managers get per year?"

    Bad search_query:
    "policies for managers: responsibilities, approvals, and requirements"

    5. Identify a policy_type ONLY when the USER QUESTION clearly
    refers to one of the available Policy Types.

    6. Identify a department ONLY when the USER QUESTION explicitly
    mentions or clearly specifies a department.

    7. Do NOT infer a department from the topic.

    8. If a policy type can be reliably identified, return it
    as a metadata filter.

    9. If a department can be reliably identified, return it
    as a metadata filter.

    10. Return both department and policy_type only when both can
        be determined reliably.

    11. Do NOT invent metadata values.

    12. Use ONLY the metadata values provided above.

    13. Metadata filters must exactly match the available values.

    14. If a metadata value cannot be determined reliably,
        leave that field out.

    15. Do not use:
        - source
        - policy_id
        - status
        - page

    16. Return ONLY valid JSON.

    Expected format:

    {{
        "search_query": "search query preserving the user's intent",
        "metadata_filters": {{
            "department": "HR",
            "policy_type": "Remote Work"
        }}
    }}

    If there is no metadata filter:

    {{
        "search_query": "search query preserving the user's intent",
        "metadata_filters": {{}}
    }}
    """

    response = get_response(self_query_prompt)

    try:
        # Remove possible markdown code fences
        response = response.strip()

        if response.startswith("```"):
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response = response.strip()

        result = json.loads(response)

        return result

    except json.JSONDecodeError:

        print("\nSelf-query parsing failed.")
        print("LLM response:")
        print(response)

        # Safe fallback
        return {"search_query": user_query, "metadata_filters": {}}


def filter_documents_by_metadata(documents, metadata_filters):
    """
    Filters BM25 documents using the metadata generated
    by the Self-Query layer.
    """

    if not metadata_filters:
        return documents

    filtered_documents = []

    for doc in documents:

        match = True

        for field, value in metadata_filters.items():

            if doc.metadata.get(field) != value:
                match = False
                break

        if match:
            filtered_documents.append(doc)

    return filtered_documents

#  RRF FUNCTION

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):

    scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[doc_id] for doc_id, score in ranked_docs]

def check_query_scope(query):
    scope_prompt = f"""
    You are a scope classifier for XYZTech's company policy assistant.

    Determine whether the user's question is related to:
    - XYZTech company policies
    - employees
    - HR
    - workplace procedures
    - leave
    - remote work
    - international work
    - travel and expenses
    - company assets
    - IT policies
    - employment practices
    - offboarding
    - workplace compliance
    - other company-policy matters

    Return exactly ONE word:

    IN_SCOPE
    or
    OUT_OF_SCOPE

    Examples:

    "What is the capital of France?"
    OUT_OF_SCOPE

    "How many sick leave days do employees get?"
    IN_SCOPE

    "Can I work from Dubai for two months?"
    IN_SCOPE

    "How do I make biryani?"
    OUT_OF_SCOPE

    "Who is responsible for asset management?"
    IN_SCOPE

    User question:
    {query}
    """

    result = get_response(scope_prompt).strip().upper()

    if "OUT_OF_SCOPE" in result:
        return "OUT_OF_SCOPE"

    return "IN_SCOPE"


# ============================================================
# MAIN RAG FUNCTION
# ============================================================


def get_rag_response(query, conversation_history):

    print("\n========== RAG START ==========", flush=True)
    print("QUERY:", query, flush=True)
    print("HISTORY LENGTH:", len(conversation_history), flush=True)

    # ==========================================
    # CONVERSATION-AWARE QUERY REWRITING
    # ==========================================

    print("BEFORE REWRITE", flush=True)

    rewritten_query = query

    if conversation_history:

        conversation_text = ""

        for message in conversation_history[-6:]:
            role = message.get("role")
            content = message.get("content", "")

            if role in ["user", "assistant"]:
                conversation_text += f"{role.upper()}: {content}\n"

        rewrite_prompt = f"""
    You are helping a company policy RAG system.

    Your job is to rewrite the user's latest question into a
    standalone search query ONLY when the latest question depends
    on the previous conversation.

    Previous conversation:
    {conversation_text}

    Latest user question:
    {query}

    Rules:

    1. If the latest question is already clear and standalone,
    return it unchanged.

    2. If the latest question depends on previous conversation,
    rewrite it using the relevant context.

    3. Preserve the user's actual intent.

    4. Do NOT answer the question.

    5. Do NOT add information that is not present in the conversation.

    6. Return ONLY the rewritten search query.

    Rewritten search query:
    """

        rewritten_query = get_response(rewrite_prompt).strip()

        print("After REWRITE", flush=True)

        if not rewritten_query:
            rewritten_query = query


    print("\n==============================")
    print("CONVERSATION QUERY REWRITE")
    print("==============================")
    print("Original Query:", query)
    print("Rewritten Query:", rewritten_query)

    # ==========================================
    # SCOPE CHECK
    # ==========================================
    print("BEFORE Scope", flush=True)
    
    scope = check_query_scope(query)
    
    if scope == "OUT_OF_SCOPE":
        response = """OUT OF SCOPE
    
        ANSWER:
        I can help with questions related to XYZTech company policies, employee guidelines, and workplace procedures."""
    
        return response, [], False


    # ==========================================
    # METADATA-AWARE QUERY
    # ==========================================
    print("BEFORE self query", flush=True)

    self_query_result = generate_self_query(rewritten_query)

    self_search_query = self_query_result["search_query"]
    metadata_filters = self_query_result["metadata_filters"]


    print("\n==============================")
    print("SELF QUERY")
    print("==============================")
    print("Original User Query:", query)
    print("Conversation-Rewritten Query:", rewritten_query)
    print("Self-Query Search Query:", self_search_query)
    print("Metadata Filters:", metadata_filters)

    print("After self query", flush=True)

    # VECTOR SEARCH

    if metadata_filters:
        print("BUILDING CHROMA FILTER", flush=True)
        conditions = [ {field: value} for field, value in metadata_filters.items() ]
        print("CHROMA CONDITIONS BUILT:", conditions, flush=True)

        if len(conditions) == 1: 
            chroma_filter = conditions[0] 
        else: 
            chroma_filter = {"$and": conditions}
    else:
        chroma_filter = None

    print("CHROMA FILTER:", chroma_filter, flush=True)

    print("STARTING VECTOR SEARCH")

    vector_results = db.max_marginal_relevance_search(
        self_search_query, k=5, fetch_k=10, filter=chroma_filter
    )
    print("VECTOR SEARCH COMPLETE")

    # BM25 SEARCH
    print("BEFORE BM25", flush=True)

    filtered_documents = filter_documents_by_metadata(documents, metadata_filters)
    print("FILTERING COMPLETE")

    if filtered_documents:
        filtered_bm25_retriever = BM25Retriever.from_documents(filtered_documents)

        filtered_bm25_retriever.k = 5
        print("BM25 BUILT")

        bm25_results = filtered_bm25_retriever.invoke(self_search_query)
        print("BM25 SEARCH COMPLETE")

    else:
        bm25_results = []

    # HYBRID SEARCH USING RRF

    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)

    # Keep top 5 results
    hybrid_results = hybrid_results[:5]

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print("\n==============================")
    print("VECTOR SEARCH RESULTS")
    print("==============================")

    for i, doc in enumerate(vector_results, start=1):
        print(f"\n--- Vector Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:500])

    print("\n==============================")
    print("BM25 SEARCH RESULTS")
    print("==============================")

    for i, doc in enumerate(bm25_results, start=1):
        print(f"\n--- BM25 Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:300])

    print("\n==============================")
    print("FINAL HYBRID RESULTS")
    print("==============================")

    for i, doc in enumerate(hybrid_results, start=1):
        print(f"\n--- Hybrid Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:300])

    if not hybrid_results:

        return ("I couldn't find this information in the policy documents.", [], False )
    # ==========================================
    # BUILD POLICY CONTEXT
    # ==========================================

    context = ""

    for i, doc in enumerate(hybrid_results, start=1):

        context += f"""
    --- POLICY CHUNK {i} ---

    Policy Content:
    {doc.page_content}

    Policy Metadata:
    Title: {doc.metadata.get("title")}
    Department: {doc.metadata.get("department")}
    Policy Type: {doc.metadata.get("policy_type")}
    Policy ID: {doc.metadata.get("policy_id")}
    Version: {doc.metadata.get("version")}
    Effective Date: {doc.metadata.get("effective_date")}
    Status: {doc.metadata.get("status")}
    Source: {doc.metadata.get("source")}
    Page: {doc.metadata.get("page")}

    """
    # ==========================================
    # 3. CREATE GROUNDED PROMPT
    # ==========================================
    answer_prompt = f"""
    You are PolicyLens AI, an AI company policy assistant.

    Your task is to answer the user's question using the provided company policy context.
    If the user describes a specific situation and asks whether an action is allowed,
    then perform a compliance analysis.
    IMPORTANT RULES:

    1. Use ONLY the information provided in the policy context.
    2. Do NOT use outside knowledge.
    3. Do NOT invent policy requirements.
    4. Determine exactly ONE response type:
        - INFORMATIONAL
        - COMPLIANT
        - NON-COMPLIANT
        - NEEDS REVIEW
        - OUT OF SCOPE

    5. FIRST DETERMINE THE USER'S INTENT.

        There are two main types of questions:

        1. INFORMATIONAL
        The user is asking for a fact, explanation, process, responsibility,
        entitlement, contact information, or other general information.

        2. COMPLIANCE
        The user describes a specific action or situation and asks whether
        it is allowed, prohibited, or compliant with company policy.

        Do not require a personal situation for an INFORMATIONAL question.

        Examples:

        "What is my HR manager's phone number?"
        → INFORMATIONAL

        "Who is responsible for asset management?"
        → INFORMATIONAL

        "How many sick leave days do I get?"
        → INFORMATIONAL

        "Can I work from Dubai for two months?"
        → COMPLIANCE


    6. INFORMATIONAL questions should be answered conversationally.

        Examples:

        "How many sick leave days can I take?"
        → INFORMATIONAL

        "How many annual leave days do I get?"
        → INFORMATIONAL

        "Can sick leave be taken in hours?"
        → INFORMATIONAL

        For INFORMATIONAL questions, do not add compliance analysis simply because the user's personal situation is unknown.

        INFORMATIONAL should be used when the user is asking for general information, whether or not the requested 
        information is present in the policy context. If the information is not present, clearly state that 
        it is not provided in the available policy documents.

    7. Use COMPLIANT only when the user describes a specific action,request, or situation and 
        the policy clearly allows that situation with all required conditions satisfied.      

    8. Use NON-COMPLIANT only when the user describes a specific action,request, or situation 
        that clearly violates an explicit policy requirement.

    10. Use NEEDS REVIEW when:

        - the user describes a specific situation but the policy does not provide enough information to make a definite compliance decision, OR
        - the policy explicitly requires additional approval, legal, tax, regulatory, or case-specific review.

    11. Do NOT treat missing information as proof of compliance.

    12. If the user is asking only for factual information that is clearly available in the policy, do NOT 
        return NEEDS REVIEW just because the user's personal circumstances are unknown.

    13. For INFORMATIONAL questions, answer the question directly. Do not create a compliance conclusion.

    14. Keep the response concise and easy to read. Do NOT repeat the same information unnecessarily across ANSWER,
        WHY, and WHAT YOU CAN DO.

    15. Formatting inside each field:
        - Use **bold** for important requirements, conditions, restrictions,approvals, or conclusions.
        - Use bullet points (as markdown "- " lines) when listing multiple requirements.
        - Use short sentences, not long blocks of text.

    16. Do NOT include policy metadata such as: policy ID, version,effective date, status, source, 
        page number, or file name in any field. This information is attached separately by the application
        from the retrieved document, not by you.

    17. Do NOT create a Sources, Policy Reference, or Metadata section.

    18. For COMPLIANT, NON-COMPLIANT, or NEEDS REVIEW responses, SUMMARY must be exactly one plain sentence, 
        in plain language, with no markdown or bold. It should state the bottom line only.

    19. IMPORTANT: DISTINGUISH OUT OF SCOPE FROM MISSING INFORMATION

        If the user's question is related to XYZTech, employees, HR, workplace
        policies, procedures, leave, remote work, assets, international work,
        or another company-policy topic, it is IN-SCOPE.

        If the question is IN-SCOPE but the requested information is not present
        in the policy context:

        - Return INFORMATIONAL.
        - Answer the user's actual question directly.
        - Clearly state that the available policy documents do not provide
        the requested information.
        - Do not invent, guess, or use outside knowledge.
        - Do not ask the user to provide a personal situation.
        - Do not treat unrelated retrieved content as evidence.

        OUT OF SCOPE should be used only when the question is clearly unrelated
        to the company-policy domain.


    20. OUT OF SCOPE

        Use OUT OF SCOPE only when the user's question is clearly unrelated to the company-policy domain.

        Examples:
        - "What is the capital of France?"
        - "Who won the cricket match?"
        - "How do I make biryani?"
        - "What is today's weather?"
        - "What is Python?"
        - "What is the population of India?"

        For OUT OF SCOPE questions:

        - First output exactly:
        OUT OF SCOPE

        - Then return exactly this structure:

        ANSWER:
        <Briefly state that the assistant only handles company-policy questions.>

        - Do NOT use information from the retrieved policy context.
        - Do NOT say that the answer was not found in the policy documents.
        - Do NOT provide the actual answer to the unrelated question.
        - Do NOT include WHY or WHAT YOU CAN DO sections.
        - Keep the response to 1-2 short sentences.

        Example:

        User question:
        "What is the capital of France?"

        Correct response:

        OUT OF SCOPE

        ANSWER:
        I can help with questions related to XYZTech company policies, employee guidelines, and workplace procedures.

    21. For INFORMATIONAL questions:

        Use INFORMATIONAL when the user is asking for information or a general policy fact. 
        The user does NOT need to describe a personal situation.

        Examples:
        - "How many sick leave days do I get?"
        - "Who is responsible for asset management?"
        - "What is my HR manager's phone number?"
        - "What is the process for returning company assets?"

        If the requested information is clearly present in the policy context:
        - Answer the user's question directly.
        - Use only the relevant policy information.

        If the requested information is NOT present in the policy context:
        - Still return INFORMATIONAL.
        - Clearly state that the available policy documents do not provide the requested information.
        - Do not guess, invent, or use outside knowledge.
        - Do not ask the user to provide a personal situation.
        - Do not say that the user has not provided a question.

        For INFORMATIONAL responses, use exactly this structure:

        INFORMATIONAL

        ANSWER:
        <Direct answer to the user's question.>

        WHY:
        <Brief explanation based only on the policy context.>

        WHAT YOU CAN DO:
        <Practical next step, if applicable. If no action is needed, write:
        "No additional action is required.">

        Keep all sections concise.
        Do not perform compliance analysis.
        Do not use COMPLIANT, NON-COMPLIANT, or NEEDS REVIEW for an
        INFORMATIONAL question.

    22. For COMPLIANT, NON-COMPLIANT, or NEEDS REVIEW:

        First, output exactly ONE of these response types on its own line:

        COMPLIANT
        NON-COMPLIANT
        NEEDS REVIEW

        Then return the answer in exactly this structure:

        SUMMARY:
        <One plain sentence stating the bottom line. No markdown.>

        REASON:
        <Explain why the situation has that compliance status using only the policy context.>

        RECOMMENDATION:
        <State the appropriate next step based only on the policy.
        Omit this section if no practical next step is needed.>

        POLICY REQUIREMENT:
        <Explain only the policy requirements directly relevant to the user's question or situation.
        Use bullet points when there are multiple requirements.
        Bold the most important requirements.>

        IMPORTANT:
        Classify the USER QUESTION first.
        Do not assume that retrieved POLICY CONTEXT makes an unrelated question in-scope.

    USER SITUATION:
    {query}

    POLICY CONTEXT:
    {context}
    """
    # ==========================================
    # 4. SEND CONTEXT + QUESTION TO LLM
    # ==========================================

    response = get_response(answer_prompt)

    # ==========================================
    # 5. DETERMINE WHETHER TO SHOW SOURCES
    # ==========================================

    response_upper = response.upper() 

    if response_upper.startswith("OUT OF SCOPE"):
        show_sources = False 

    elif response_upper.startswith("INFORMATIONAL"): 
        missing_information_phrases = [ 
            "DO NOT SPECIFY", 
            "DOES NOT SPECIFY", 
            "NOT SPECIFIED", 
            "NOT PROVIDED", 
            "DO NOT PROVIDE", 
            "DOES NOT PROVIDE", 
            "NOT CONTAIN", 
            "DOES NOT CONTAIN", 
            "DO NOT INCLUDE",
            "DOES NOT INCLUDE",
            "NOT AVAILABLE IN THE POLICY",
            "NOT AVAILABLE IN THE PROVIDED POLICY DOCUMENTS",
            "NOT FOUND IN THE POLICY",
            "NOT FOUND IN THE PROVIDED POLICY DOCUMENTS",
            ] 

        show_sources = not any( 
            phrase in response_upper 
            for phrase in missing_information_phrases 
            ) 

    else: 
    # COMPLIANT, NON-COMPLIANT, NEEDS REVIEW 
        show_sources = True

    # ========================================== 
    #  6. RETURN ANSWER + SOURCES 
    # ==========================================

    sources = []

    for doc in hybrid_results:

        sources.append(
            {
                "source": doc.metadata.get("source"),
                "policy_id": doc.metadata.get("policy_id"),
                "policy_type": doc.metadata.get("policy_type"),
                "version": doc.metadata.get("version"),
                "effective_date": doc.metadata.get("effective_date"),
                "status": doc.metadata.get("status"),
                "page": doc.metadata.get("page"),
            }
        )

    return response, sources, show_sources
