'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys
import time  # For time.time()
import json  # For Improvement 1 & 3
import re
from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = float(config_data["LLM_TEMPERATURE"]) # Cast to float to prevent TypeError
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False
# Add this to your main code before running Mode 30:
print("\n=== VECTOR DB DIAGNOSTIC ===")
test_search = vectorstore.similarity_search_with_score("psoriasis", k=5)
print(f"Test search for 'psoriasis' returned {len(test_search)} results")
for i, (doc, score) in enumerate(test_search[:3]):
    print(f"  Result {i+1}: {doc.page_content[:50]}... (score: {score:.4f})")

test_search2 = vectorstore.similarity_search_with_score("HLA-B", k=5)
print(f"Test search for 'HLA-B' returned {len(test_search2)} results")
for i, (doc, score) in enumerate(test_search2[:3]):
    print(f"  Result {i+1}: {doc.page_content[:50]}... (score: {score:.4f})")
print("=========================\n")

print(vectorstore)

import os

print(f"VECTOR_DB_PATH: {VECTOR_DB_PATH}")
print(f"Path exists: {os.path.exists(VECTOR_DB_PATH)}")
if os.path.exists(VECTOR_DB_PATH):
    print(f"Contents: {os.listdir(VECTOR_DB_PATH)}")
    # Check size
    total_size = sum(os.path.getsize(os.path.join(VECTOR_DB_PATH, f)) 
                     for f in os.listdir(VECTOR_DB_PATH) if os.path.isfile(os.path.join(VECTOR_DB_PATH, f)))
    print(f"Total size: {total_size / (1024*1024):.2f} MB")

print(f"\nnode_context_df info:")
print(f"  Shape: {node_context_df.shape}")
print(f"  Columns: {node_context_df.columns.tolist()}")
if not node_context_df.empty:
    print(f"  Sample rows:")
    print(node_context_df.head(3))
    print(f"\n  Sample node names:")
    print(node_context_df['node_name'].head(10).tolist())
else:
    print("  ⚠️ DataFrame is EMPTY!")


# ==============================================================================
#                 !!! IMPORTANT: SET THIS FOR EACH RUN !!!
# ==============================================================================
MODE = "30"  # <-- This is set to "5" for your bonus experiment
# ==============================================================================

### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 
### MODE 4 -30: Enhanced Bonus 					###

# --- New System Prompt for the Filter LLM (for Mode 4) ---
FILTER_SYSTEM_PROMPT = """You are an expert biomedical data filter. Your task is to re-write the provided context, keeping ONLY gene-disease associations.
You MUST remove all other information, especially text about 'Provenance', 'sources', 'symptoms', 'identifiers', or any other metadata.
Output only the clean facts.

Example:
INPUT: 'Disease psoriasis associates Gene SLC29A3 and Provenance of this association is HPO. Disease polyarteritis nodosa associates Gene SLC29A3 and Provenance of this association is NCBI PubMed.'
OUTPUT: 'Disease psoriasis associates Gene SLC29A3. Disease polyarteritis nodosa associates Gene SLC29A3.'
"""

# --- New System Prompt for Few-Shot CoT (for Mode 5) ---
BONUS_SYSTEM_PROMPT = """You are an expert biomedical researcher. Your task is to answer the multiple-choice question based ONLY on the provided context.

You must first show your reasoning step-by-step in a 'Thinking:' block.
In your thinking, you must:
1. Identify the diseases in the question.
2. Scan the context for gene associations for *each* disease.
3. Explicitly state that you are IGNORING irrelevant information like 'Provenance' or 'symptoms'.
4. Find the common gene(s) that appear for all diseases.
5. Check the "Given list" to find your answer.

Finally, provide your answer in a JSON object with a single 'answer' key.

---
[EXAMPLE]

Context: Disease psoriasis associates Gene SLC29A3 and Provenance of this association is HPO. Disease polyarteritis nodosa associates Gene SLC29A3 and Provenance of this association is NCBI PubMed.
Question: Out of the given list, which Gene is associated with psoriasis and polyarteritis nodosa. Given list is: SHTN1, HLA-B, SLC29A3, DTNB.

Thinking:
1. The question asks for a single gene associated with both 'psoriasis' and 'polyarteritis nodosa'.
2. I will scan the context for these two diseases. I will ignore all 'Provenance' information as it is irrelevant.
3. Context evidence for 'psoriasis': "Disease psoriasis associates Gene SLC29A3".
4. Context evidence for 'polyarteritis nodosa': "Disease polyarteritis nodosa associates Gene SLC29A3".
5. The common gene associated with both diseases is 'SLC29A3'.
6. The 'Given list' is SHTN1, HLA-B, SLC29A3, DTNB.
7. 'SLC29A3' is present in the list.
8. Therefore, the correct answer is 'SLC29A3'.

Final Answer:
{
  "answer": "SLC29A3"
}
---
"""
def retrieve_context_from_csv(question, node_context_df, embedding_function, context_volume, 
                              context_sim_threshold, context_sim_min_threshold, model_id):
    """
    Direct retrieval from CSV - bypasses vectorstore completely
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Extract disease entities from question
    entities = disease_entity_extractor_v2(question, model_id)
    
    if not entities:
        # No entities found, return empty
        return ""
    
    # Match entities to node names in CSV
    node_hits = []
    for entity in entities:
        entity_lower = entity.lower().strip()
        
        # Try exact match first
        exact_match = node_context_df[node_context_df['node_name'].str.lower() == entity_lower]
        if not exact_match.empty:
            node_hits.append(exact_match.iloc[0]['node_name'])
            continue
        
        # Try substring match (contains)
        substring_match = node_context_df[node_context_df['node_name'].str.lower().str.contains(entity_lower, na=False, regex=False)]
        if not substring_match.empty:
            # Take the shortest match (most specific)
            best_match = substring_match.loc[substring_match['node_name'].str.len().idxmin()]
            node_hits.append(best_match['node_name'])
            continue
    
    if not node_hits:
        return ""
    
    # Extract and rank context for matched nodes
    question_embedding = embedding_function.embed_query(question)
    max_context_per_node = max(1, int(context_volume / len(node_hits)))
    
    node_context_extracted = ""
    for node_name in node_hits:
        # Get context from CSV
        node_context_row = node_context_df[node_context_df.node_name == node_name]
        if node_context_row.empty:
            continue
            
        node_context = node_context_row.node_context.values[0]
        
        # Split context into sentences
        node_context_list = [s.strip() for s in node_context.split(". ") if len(s.strip()) > 10]
        
        if not node_context_list:
            continue
        
        # Embed context sentences
        node_context_embeddings = embedding_function.embed_documents(node_context_list)
        
        # Calculate similarity scores
        similarities = []
        for node_context_embedding in node_context_embeddings:
            sim = cosine_similarity(
                np.array(question_embedding).reshape(1, -1), 
                np.array(node_context_embedding).reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Rank by similarity
        similarities_indexed = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        
        # Apply thresholds
        if similarities:
            percentile_threshold = np.percentile(similarities, context_sim_threshold)
            high_similarity_indices = [
                idx for idx, sim in similarities_indexed 
                if sim > max(percentile_threshold, context_sim_min_threshold)
            ][:max_context_per_node]
            
            # Extract high-similarity context
            high_similarity_context = [node_context_list[idx] for idx in high_similarity_indices]
            
            if high_similarity_context:
                node_context_extracted += ". ".join(high_similarity_context) + ". "
    
    return node_context_extracted.strip()


def mode_30_ultimate_pipeline(row, question, vectorstore, embedding_function_for_context_retrieval, 
                               node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                               QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, CHAT_MODEL_ID,
                               get_Gemini_response, SYSTEM_PROMPT, TEMPERATURE, retrieve_context):
    """
    Ultimate multi-stage pipeline for biomedical MCQ answering - CSV-based retrieval
    """
    
    import re
    from collections import Counter
    
    RETRIEVAL_MODEL_ID = "models/gemini-flash-latest"
    
    # ========================================================================
    # STAGE 1: INTELLIGENT QUERY DECOMPOSITION
    # ========================================================================
    
    diseases_match = re.search(r'associated with (.+?) and (.+?)\. Given', question)
    gene_list_match = re.search(r'Given list is:(.+)', question)
    
    diseases = []
    gene_options = []
    
    if diseases_match:
        diseases = [diseases_match.group(1).strip(), diseases_match.group(2).strip()]
    
    if gene_list_match:
        genes_text = gene_list_match.group(1)
        gene_options = [g.strip() for g in genes_text.split(',') if g.strip()]
    
    # ========================================================================
    # STAGE 2: MULTI-STRATEGY RETRIEVAL (CSV-based)
    # ========================================================================
    
    contexts = {}
    
    print(f"    [Mode 30] Stage 1: Decomposed into {len(diseases)} diseases, {len(gene_options)} gene options")
    print(f"    [Mode 30] Stage 2: Multi-strategy retrieval (CSV-based)...")
    
    # Strategy 1: Standard retrieval on full question
    try:
        contexts['full_question'] = retrieve_context_from_csv(
            question, node_context_df, embedding_function_for_context_retrieval,
            CONTEXT_VOLUME, 
            QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
            QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
            model_id=RETRIEVAL_MODEL_ID
        )
        if contexts['full_question'] and len(contexts['full_question']) > 30:
            print(f"      ✓ Strategy 1 (full question): {len(contexts['full_question'])} chars")
        else:
            print(f"      ⚠ Strategy 1: empty context")
            contexts['full_question'] = ""
    except Exception as e:
        contexts['full_question'] = ""
        print(f"      ✗ Strategy 1 failed: {e}")
    
    # Strategy 2: Disease-specific retrieval
    if diseases:
        for i, disease in enumerate(diseases):
            try:
                contexts[f'disease_{i+1}'] = retrieve_context_from_csv(
                    disease, node_context_df, embedding_function_for_context_retrieval,
                    CONTEXT_VOLUME // 2,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,
                    model_id=RETRIEVAL_MODEL_ID
                )
                if contexts[f'disease_{i+1}'] and len(contexts[f'disease_{i+1}']) > 30:
                    print(f"      ✓ Strategy 2.{i+1} ({disease[:30]}...): {len(contexts[f'disease_{i+1}'])} chars")
                else:
                    print(f"      ⚠ Strategy 2.{i+1}: empty context")
                    contexts[f'disease_{i+1}'] = ""
            except Exception as e:
                contexts[f'disease_{i+1}'] = ""
                print(f"      ✗ Strategy 2.{i+1} failed: {e}")
    
    # Strategy 3: Gene-specific retrieval (for top candidates)
    if gene_options:
        for i, gene in enumerate(gene_options[:3]):
            try:
                contexts[f'gene_{gene}'] = retrieve_context_from_csv(
                    f"{gene} disease association", node_context_df, 
                    embedding_function_for_context_retrieval,
                    CONTEXT_VOLUME // 3,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.6,
                    model_id=RETRIEVAL_MODEL_ID
                )
                if contexts[f'gene_{gene}'] and len(contexts[f'gene_{gene}']) > 30:
                    print(f"      ✓ Strategy 3.{i+1} ({gene}): {len(contexts[f'gene_{gene}'])} chars")
                else:
                    print(f"      ⚠ Strategy 3.{i+1}: empty context")
                    contexts[f'gene_{gene}'] = ""
            except Exception as e:
                contexts[f'gene_{gene}'] = ""
                print(f"      ✗ Strategy 3.{i+1} failed: {e}")
    
    # Strategy 4: Combined disease query
    if len(diseases) >= 2:
        try:
            combined_disease_query = f"{diseases[0]} {diseases[1]} shared genetic basis"
            contexts['disease_combined'] = retrieve_context_from_csv(
                combined_disease_query, node_context_df, 
                embedding_function_for_context_retrieval,
                CONTEXT_VOLUME,
                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.75,
                model_id=RETRIEVAL_MODEL_ID
            )
            if contexts['disease_combined'] and len(contexts['disease_combined']) > 30:
                print(f"      ✓ Strategy 4 (combined): {len(contexts['disease_combined'])} chars")
            else:
                print(f"      ⚠ Strategy 4: empty context")
                contexts['disease_combined'] = ""
        except Exception as e:
            contexts['disease_combined'] = ""
            print(f"      ✗ Strategy 4 failed: {e}")
    
    # ========================================================================
    # STAGE 3: CONTEXT QUALITY ASSESSMENT & AGGREGATION
    # ========================================================================
    
    def assess_context_quality(ctx):
        if not ctx or len(ctx.strip()) < 30:
            return 0
        score = len(ctx)
        for gene in gene_options[:5]:
            if gene.lower() in ctx.lower():
                score += 100
        for disease in diseases:
            if disease.lower() in ctx.lower():
                score += 50
        return score
    
    context_scores = {k: assess_context_quality(v) for k, v in contexts.items()}
    sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
    top_contexts = [contexts[k] for k, score in sorted_contexts[:3] if score > 0]
    
    if top_contexts:
        aggregated_context = "\n\n=== RETRIEVED CONTEXT ===\n".join(top_contexts)
        best_score = context_scores[sorted_contexts[0][0]]
        context_quality = "HIGH" if best_score > 500 else ("MEDIUM" if best_score > 200 else "LOW")
    else:
        aggregated_context = ""
        context_quality = "NONE"
    
    successful_retrievals = len([v for v in contexts.values() if v and len(v) > 30])
    print(f"    [Mode 30] Stage 3: Context quality = {context_quality}, using {len(top_contexts)} sources")
    print(f"              Successful retrievals: {successful_retrievals}/{len(contexts)}")
    
    # ========================================================================
    # STAGE 4: ADAPTIVE REASONING
    # ========================================================================
    
    if context_quality in ["HIGH", "MEDIUM"]:
        reasoning_framework = {
            "context_quality": f"{context_quality} - Context available from {len(top_contexts)} sources",
            "retrieved_context": aggregated_context,
            "filtering_rules": {
                "IGNORE": ["provenance information", "data source metadata", "symptom-only descriptions"],
                "PRIORITIZE": ["explicit gene-disease associations", "molecular mechanisms", "protein interactions"]
            },
            "evaluation_protocol": [
                "Step 1: Extract all gene options from the question",
                "Step 2: For EACH gene option, search the retrieved context for associations with BOTH diseases",
                "Step 3: Score each gene: 3 pts (both diseases), 2 pts (one + mechanism), 1 pt (weak), 0 pts (none)",
                "Step 4: Select the gene with HIGHEST total score",
                "Step 5: Verify evidence in context"
            ],
            "critical_rule": "TRUST THE CONTEXT. Select based on evidence, even if surprising."
        }
    elif context_quality == "LOW":
        reasoning_framework = {
            "context_quality": "LOW - Minimal context available",
            "retrieved_context": aggregated_context,
            "evaluation_protocol": [
                "Step 1: Check context for ANY gene mentions",
                "Step 2: Supplement with biomedical knowledge",
                "Step 3: Prioritize HLA genes for autoimmune, TERT for cancer, pleiotropic genes",
                "Step 4: Select most plausible based on combined evidence"
            ],
            "guidance": "Use sparse context + established medical genetics principles"
        }
    else:  # NONE
        reasoning_framework = {
            "context_quality": "NONE - No relevant context retrieved",
            "fallback_mode": "Using established biomedical knowledge only",
            "knowledge_based_strategy": [
                "1. HLA genes (HLA-B, HLA-DQA1, etc.) → autoimmune/inflammatory diseases",
                "2. TERT → multiple cancers",
                "3. Pleiotropic genes bridge unrelated diseases",
                "4. Gene families share disease associations",
                "5. Checkpoint genes (CHEK2, ATM) → cancer predisposition"
            ],
            "evaluation_protocol": [
                "Step 1: Classify both diseases (autoimmune? cancer? metabolic?)",
                "Step 2: Recall known disease associations for each gene option",
                "Step 3: Identify multi-system genes",
                "Step 4: Select most biologically plausible"
            ],
            "warning": "LOW CONFIDENCE - Answer based on general knowledge without graph support"
        }
    
    # ========================================================================
    # STAGE 5: MULTI-TEMPERATURE ENSEMBLE
    # ========================================================================
    
    print(f"    [Mode 30] Stage 4: Using {context_quality} quality reasoning")
    print(f"    [Mode 30] Stage 5: Running ensemble with 3 temperatures...")
    
    json_string_context = json.dumps(reasoning_framework, indent=2)
    
    enriched_prompt = f"""Biomedical Knowledge Analysis Framework:
{json_string_context}

Question: {question}

CRITICAL INSTRUCTIONS:
1. Evaluate EVERY gene option listed in the question
2. Show your scoring/reasoning clearly
3. Select the option with strongest evidence or biological plausibility
4. Answer MUST be exactly one gene name from the given list
5. Format: {{"answer": "GENE_NAME"}}
"""
    
    answers = []
    
    for temp in [0.0, 0.3, 0.5]:
        try:
            output_temp = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=temp)
            match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_temp))
            if match:
                candidate = match.group(1).strip()
                if not gene_options or candidate in gene_options or any(candidate in opt for opt in gene_options):
                    answers.append(candidate)
                    print(f"      ✓ Temp {temp}: {candidate}")
        except Exception as e:
            print(f"      ✗ Temp {temp} failed: {e}")
    
    # ========================================================================
    # STAGE 6: VOTING
    # ========================================================================
    
    print(f"    [Mode 30] Stage 6: Voting among {len(answers)} answers...")
    
    if len(answers) >= 2:
        vote_counts = Counter(answers)
        if vote_counts.most_common(1)[0][1] >= 2:
            final_answer = vote_counts.most_common(1)[0][0]
            confidence = "HIGH"
            agreement = vote_counts.most_common(1)[0][1]
        else:
            final_answer = answers[0] if answers else None
            confidence = "MEDIUM"
            agreement = 1
    elif len(answers) == 1:
        final_answer = answers[0]
        confidence = "MEDIUM"
        agreement = 1
    else:
        final_answer = None
        confidence = "LOW"
        agreement = 0
    
    print(f"    [Mode 30] Final: {final_answer} (confidence: {confidence}, agreement: {agreement}/{len(answers)})")
    
    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    
    output = json.dumps({
        "answer": final_answer,
        "confidence": confidence,
        "context_quality": context_quality,
        "vote_agreement": f"{agreement}/{len(answers)}",
        "retrieval_strategies_used": successful_retrievals,
        "note": "CSV-based multi-stage adaptive retrieval + ensemble reasoning"
    })
    
    return output

# Tried with different retrieval techniques.

def retrieve_context_direct(question, node_context_df, embedding_function, context_volume, 
                           context_sim_threshold, context_sim_min_threshold, model_id, api=False):
    """
    Direct retrieval bypassing vectorstore - matches against node_context_df directly
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    print(f"    [Direct Retrieval] question: {question}")
    
    # Extract entities
    entities = disease_entity_extractor_v2(question, model_id)
    print(f"    [Direct Retrieval] entities: {entities}")
    
    if not entities:
        print(f"    [Direct Retrieval] No entities found, returning empty")
        return ""
    
    # Match entities to node names using fuzzy matching
    node_hits = []
    for entity in entities:
        entity_lower = entity.lower().strip()
        
        # Try exact match first
        exact_match = node_context_df[node_context_df['node_name'].str.lower() == entity_lower]
        if not exact_match.empty:
            node_hits.append(exact_match.iloc[0]['node_name'])
            print(f"    [Direct Retrieval] Exact match for '{entity}': {exact_match.iloc[0]['node_name']}")
            continue
        
        # Try substring match
        substring_match = node_context_df[node_context_df['node_name'].str.lower().str.contains(entity_lower, na=False)]
        if not substring_match.empty:
            node_hits.append(substring_match.iloc[0]['node_name'])
            print(f"    [Direct Retrieval] Substring match for '{entity}': {substring_match.iloc[0]['node_name']}")
            continue
        
        # Try embedding similarity if available
        print(f"    [Direct Retrieval] No match found for '{entity}'")
    
    if not node_hits:
        print(f"    [Direct Retrieval] No node hits found")
        return ""
    
    # Now extract context for matched nodes
    question_embedding = embedding_function.embed_query(question)
    max_context_per_node = int(context_volume / len(node_hits))
    
    node_context_extracted = ""
    for node_name in node_hits:
        if api:
            # Use API to get context
            try:
                node_context, context_table = get_context_using_spoke_api(node_name)
                print(f"    [Direct Retrieval] API fetched {len(node_context)} chars for '{node_name}'")
            except Exception as e:
                print(f"    [Direct Retrieval] API failed for '{node_name}': {e}")
                # Fallback to CSV
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        else:
            # Use CSV context
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        
        # Split and embed context sentences
        node_context_list = node_context.split(". ")
        if not node_context_list:
            continue
            
        node_context_embeddings = embedding_function.embed_documents(node_context_list)
        
        # Calculate similarities
        similarities = [
            cosine_similarity(
                np.array(question_embedding).reshape(1, -1), 
                np.array(node_context_embedding).reshape(1, -1)
            )[0][0]
            for node_context_embedding in node_context_embeddings
        ]
        
        # Find high similarity context
        similarities_indexed = sorted([(s, i) for i, s in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s for s, _ in similarities_indexed], context_sim_threshold)
        
        high_similarity_indices = [
            i for s, i in similarities_indexed 
            if s > percentile_threshold and s > context_sim_min_threshold
        ][:max_context_per_node]
        
        high_similarity_context = [node_context_list[i] for i in high_similarity_indices]
        
        if high_similarity_context:
            node_context_extracted += ". ".join(high_similarity_context) + ". "
            print(f"    [Direct Retrieval] Extracted {len(high_similarity_context)} context sentences for '{node_name}'")
    
    print(f"    [Direct Retrieval] Total context: {len(node_context_extracted)} chars")
    return node_context_extracted


def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)

    # Set this to 0 to run all questions
    START_INDEX = 0 
    question_df = question_df.iloc[START_INDEX:]

    answer_list = []
    
    # Define and create the new save file
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    
    # If starting fresh, create a new file with headers
    if START_INDEX == 0:
        pd.DataFrame(columns=["question", "correct_answer", "llm_answer"]).to_csv(output_file, index=False, header=True)

    # Update progress bar to use the correct length
    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        try: 
            question = row["text"]
            output = "" # Initialize output
            
            if MODE != "30":
                                # Corrected function call with named arguments. This gave better results but could also use the default retrieve_context function.
                                context = retrieve_context_from_csv(
                                    question=row["text"],
                                    node_context_df=node_context_df,
                                    embedding_function=embedding_function_for_context_retrieval,
                                    context_volume=CONTEXT_VOLUME,
                                    context_sim_threshold=QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                    context_sim_min_threshold=QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                                    model_id="gemini-1.5-flash"  # This model_id is passed to disease_entity_extractor_v2
                                )
            
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ### 
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "1":
                ### MODE 1: jsonlize the context from KG search ### 
                context_json = {"retrieved_information": context}
                json_string_context = json.dumps(context_json, indent=2)
                enriched_prompt = "Context: "+ json_string_context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ### 
                prior_knowledge = "Provenance & Symptoms information is useless. Similar diseases tend to have similar gene associations."
                final_context = context + " " + prior_knowledge
                enriched_prompt = "Context: "+ final_context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            
            elif MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ### 
                prior_knowledge = [
                    "Provenance & Symptoms information is useless.",
                    "Similar diseases tend to have similar gene associations."
                ]
                context_json = {
                    "retrieved_information": context,
                    "prior_knowledge_notes": prior_knowledge
                }
                json_string_context = json.dumps(context_json, indent=2)
                enriched_prompt = "Context: "+ json_string_context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "4":
                ### MODE 4: Bonus: LLM-based Context Re-writing ###
                clean_context = "" # Initialize clean_context
                
                # Only run the Filter LLM if we actually retrieved some context
                if context and context.strip():
                    # 1. First Pass: Call the Filter LLM to clean the context
                    clean_context = get_Gemini_response(
                        instruction=context,
                        system_prompt=FILTER_SYSTEM_PROMPT,
                        temperature=TEMPERATURE
                    )
                
                # 2. Second Pass: Call the Answer LLM with the (now clean) context
                enriched_prompt = "Context: "+ clean_context + "\n" + "Question: "+ question
                output = get_Gemini_response(
                    instruction=enriched_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=TEMPERATURE
                )

            elif MODE == "6":
                ### MODE 4: Evidence-Based Structured Reasoning ###
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance information", "data source metadata", "symptom descriptions"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms", "disease relationships", "protein interactions"]
                    },
                    "reasoning_steps": [
                        "1. Extract the core question: What is being asked?",
                        "2. Identify key entities: Which diseases/genes/proteins are mentioned?",
                        "3. Filter context: Apply IGNORE and PRIORITIZE rules above",
                        "4. Find evidence: What relevant facts support each answer option?",
                        "5. Use similarity principle: Similar diseases → similar gene associations",
                        "6. Select answer: Choose the option with strongest genetic/molecular evidence"
                    ],
                    "quality_check": "Before answering, verify your reasoning uses genetic/molecular evidence, not just symptoms or provenance"
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Follow the reasoning_steps in order. Apply the filtering_rules strictly. Perform the quality_check before providing your final answer."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            elif MODE == "9":
                ### MODE 9: Adaptive Strategy Selection ###
                
                # Simple heuristic: check if question is about genes/mechanisms vs diseases
                question_lower = question.lower()
                is_gene_focused = any(word in question_lower for word in ['gene', 'protein', 'mutation', 'expression', 'pathway'])
                
                if is_gene_focused:
                    strategy = {
                        "context": context,
                        "focus": "Molecular and genetic evidence ONLY. This is a mechanism question.",
                        "ignore": "Disease symptoms, provenance"
                    }
                else:
                    strategy = {
                        "context": context,
                        "focus": "Disease-gene associations and molecular similarities. Look for shared genetic patterns.",
                        "ignore": "Symptom overlap, provenance"
                    }
                
                json_string_context = json.dumps(strategy, indent=2)
                enriched_prompt = f"Context:\n{json_string_context}\n\nQuestion: {question}"
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "7":
                ### MODE 7: Contrastive Learning with Examples ###
                
                guidance = {
                    "context": context,
                    "critical_rule": "Gene associations and molecular mechanisms are reliable. Provenance and symptoms are NOT reliable for answering.",
                    "example_reasoning": {
                        "WRONG_approach": "Disease X has symptom Y, Disease Z has symptom Y, therefore they're related",
                        "CORRECT_approach": "Disease X associates with gene G1, Disease Z associates with gene G1, therefore they share molecular basis"
                    },
                    "your_approach": "Focus on genes, proteins, and molecular pathways. Ignore symptom overlap."
                }
                
                json_string_context = json.dumps(guidance, indent=2)
                enriched_prompt = f"Analysis Framework:\n{json_string_context}\n\nQuestion: {question}\n\nApply the CORRECT_approach."
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            elif MODE == "10":
                ### MODE 5: Explicit Answer Option Evaluation ###
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "data sources", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms", "disease relationships"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options from the question",
                        "Step 2: For EACH option, find supporting or contradicting evidence in retrieved_context",
                        "Step 3: Score each option based on molecular/genetic evidence strength",
                        "Step 4: Select the option with the strongest evidence match",
                        "Step 5: Double-check using the similarity principle: similar diseases → similar genes"
                    ],
                    "critical_rule": "You must explicitly consider ALL answer options before selecting one. Don't stop at the first plausible answer."
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            IMPORTANT: Evaluate EVERY answer option systematically. Show evidence for each before choosing."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "11":
                ### MODE 9: Error-Pattern Targeted Strategy ###
                
                question_lower = question.lower()
                
                # Detect question type
                is_similarity_question = any(word in question_lower for word in 
                    ['similar', 'related', 'associated', 'linked', 'connection'])
                
                if is_similarity_question:
                    # Special handling for similarity questions (if these are error-prone)
                    reasoning_framework = {
                        "context": context,
                        "similarity_focus": [
                            "Look for SHARED gene associations between diseases",
                            "Molecular pathways in common indicate true similarity",
                            "Symptom overlap alone is NOT sufficient evidence",
                            "Check for gene family relationships"
                        ],
                        "filtering": "Ignore provenance and symptom descriptions entirely"
                    }
                else:
                    # Use Mode 4's successful approach for other questions
                    reasoning_framework = {
                        "retrieved_context": context,
                        "filtering_rules": {
                            "IGNORE": ["provenance", "symptoms"],
                            "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                        },
                        "reasoning_steps": [
                            "1. Extract core question",
                            "2. Identify key entities",
                            "3. Filter using rules above",
                            "4. Find evidence for each option",
                            "5. Select strongest molecular evidence"
                        ]
                    }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                enriched_prompt = f"Context:\n{json_string_context}\n\nQuestion: {question}"
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "12":
                ### MODE 10: Adversarial Distractor Identification ###
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "data sources", "symptom descriptions"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms", "disease relationships"]
                    },
                    "distractor_awareness": {
                        "common_traps": [
                            "Options that match symptoms but not genetic evidence",
                            "Options based on name similarity without molecular basis",
                            "Options citing provenance/metadata instead of biology",
                            "Options with partial truth but missing key mechanism"
                        ],
                        "validation_checklist": [
                            "Does this option have direct gene/molecular support?",
                            "Or does it only seem right based on superficial features?",
                            "Is there a stronger option with better genetic evidence?"
                        ]
                    },
                    "evaluation_protocol": [
                        "Step 1: List all answer options",
                        "Step 2: For EACH option, identify what evidence supports it",
                        "Step 3: For EACH option, check if it matches any distractor pattern",
                        "Step 4: Rank options by strength of molecular/genetic evidence",
                        "Step 5: Flag and eliminate options that are likely distractors",
                        "Step 6: Select the option with strongest, most direct evidence"
                    ],
                    "final_check": "Before answering: Can I point to specific gene/protein/mechanism evidence? If no, reconsider."
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            CRITICAL: Watch for distractor options. Evaluate EACH option for both supporting evidence AND distractor patterns. Choose the option with the most direct molecular evidence."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "13":
                ### MODE 11: Quantitative Evidence Scoring ###
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "data sources", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms", "disease relationships"]
                    },
                    "scoring_rubric": {
                        "HIGH_SCORE_3_points": "Direct gene-disease association with mechanism explained",
                        "MEDIUM_SCORE_2_points": "Gene mentioned with disease but mechanism unclear",
                        "LOW_SCORE_1_point": "Indirect evidence or inferred relationship",
                        "ZERO_SCORE_0_points": "No molecular evidence, only symptoms or provenance"
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options from the question",
                        "Step 2: For EACH option, search for supporting evidence in retrieved_context",
                        "Step 3: Assign a score (0-3) to EACH option using the scoring_rubric",
                        "Step 4: Write down: Option A = X points (reason), Option B = Y points (reason), etc.",
                        "Step 5: Select the HIGHEST scoring option",
                        "Step 6: If tied, use similarity principle: similar diseases → similar genes"
                    ],
                    "requirement": "You MUST show the score for each option before selecting your answer"
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Score EVERY option (0-3 points) based on evidence strength. Show all scores. Select the highest."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "14":
                ### MODE 13: Question Decomposition Strategy ###
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering": "Ignore provenance and symptoms. Focus on genes, proteins, molecular mechanisms.",
                    "decomposition_protocol": [
                        "Step 1: DECOMPOSE the question into sub-questions",
                        "   - What entities are involved? (diseases, genes, proteins)",
                        "   - What relationship is being asked about? (association, similarity, mechanism)",
                        "   - What type of evidence is needed? (genetic, molecular, pathway)",
                        "Step 2: Answer each sub-question using retrieved_context",
                        "Step 3: SYNTHESIZE sub-answers to evaluate each option",
                        "Step 4: Select option that best matches ALL sub-answers"
                    ],
                    "example_decomposition": {
                        "complex_question": "Which gene is most associated with diseases similar to Disease X?",
                        "sub_questions": [
                            "Q1: What genes are associated with Disease X?",
                            "Q2: What diseases are similar to Disease X?",
                            "Q3: What genes do those similar diseases share?"
                        ],
                        "synthesis": "Answer is the gene that appears in Q1 AND Q3"
                    },
                    "principle": "Complex questions require breaking down → solving parts → combining answers"
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Context:
            {json_string_context}

            Question: {question}

            First, break this question into 2-3 sub-questions. Then answer each. Finally, combine to select the best option."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            elif MODE == "15":
                ### MODE 14: Biological First Principles Reasoning ###
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                    },
                    "biological_principles": {
                        "principle_1": "Genes in the same pathway often associate with related diseases",
                        "principle_2": "Similar diseases share molecular mechanisms, not just symptoms",
                        "principle_3": "Protein-protein interactions suggest functional relationships",
                        "principle_4": "Gene family members may have overlapping disease associations",
                        "principle_5": "Downstream effects (symptoms) don't imply upstream causes (genes)"
                    },
                    "evaluation_protocol": [
                        "1. For each answer option, identify the biological claim",
                        "2. Check: Does this claim align with biological_principles?",
                        "3. Check: Is there molecular evidence in retrieved_context?",
                        "4. Score: High if both principles + evidence support it",
                        "5. Select: Option with best principle + evidence alignment"
                    ],
                    "warning": "Avoid answers that violate biological principles even if they seem superficially plausible"
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Context:
            {json_string_context}

            Question: {question}

            Apply biological_principles to evaluate each option. Select the one that is both evidenced AND biologically sound."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "19":
                ### MODE 19: Enhanced Context Retrieval with Fallback ###
                
                # Try to get more context by adjusting retrieval parameters
                # Temporarily increase context volume for this question
                enhanced_context_volume = CONTEXT_VOLUME * 2
                
                # Lower the similarity threshold to get more results
                relaxed_similarity_threshold = QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7
                
                context = retrieve_context(
                    row["text"], 
                    vectorstore, 
                    embedding_function_for_context_retrieval, 
                    node_context_df, 
                    enhanced_context_volume,  # 2x more context
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    relaxed_similarity_threshold,  # Lower threshold
                    edge_evidence, 
                    model_id=CHAT_MODEL_ID
                )
                
                # Check if context is empty or too short
                if not context or len(context.strip()) < 50:
                    # FALLBACK: Extract ALL entities from the question including answer options
                    # Parse the "Given list is: GENE1, GENE2, ..." part
                    import re
                    gene_list_match = re.search(r'Given list is:(.+)', row["text"])
                    if gene_list_match:
                        genes_text = gene_list_match.group(1)
                        # Extract each gene/entity
                        entities = [g.strip() for g in genes_text.split(',')]
                        
                        # Try to retrieve context for EACH candidate gene
                        all_contexts = []
                        for entity in entities[:5]:  # Limit to first 5 to avoid too many calls
                            entity_context = retrieve_context(
                                entity, 
                                vectorstore, 
                                embedding_function_for_context_retrieval, 
                                node_context_df, 
                                CONTEXT_VOLUME // 2,  # Less per entity
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                relaxed_similarity_threshold,
                                edge_evidence, 
                                model_id=CHAT_MODEL_ID
                            )
                            if entity_context:
                                all_contexts.append(f"[{entity}]: {entity_context}")
                        
                        # Combine all contexts
                        if all_contexts:
                            context = "\n\n".join(all_contexts)
                
                # Now use Mode 5's strategy with better context
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "data sources", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms", "disease relationships"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options from the question",
                        "Step 2: For EACH option, find supporting or contradicting evidence in retrieved_context",
                        "Step 3: Score each option based on molecular/genetic evidence strength",
                        "Step 4: Select the option with the strongest evidence match",
                        "Step 5: Double-check using the similarity principle: similar diseases → similar genes"
                    ],
                    "critical_rule": "You must explicitly consider ALL answer options before selecting one. Don't stop at the first plausible answer."
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            IMPORTANT: Evaluate EVERY answer option systematically. Show evidence for each before choosing."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "20":
                ### MODE 20: Retrieve Context Using Question + All Answer Options ###
                
                # Extract the gene list from the question
                import re
                gene_list_match = re.search(r'Given list is:(.+)', row["text"])
                
                # Create an enhanced query that includes both the question and the answer options
                if gene_list_match:
                    genes = gene_list_match.group(1)
                    # Combine question with genes for better retrieval
                    enhanced_query = row["text"] + " " + genes
                else:
                    enhanced_query = row["text"]
                
                # Retrieve with the enhanced query
                context = retrieve_context(
                    enhanced_query,  # Include answer options in retrieval
                    vectorstore, 
                    embedding_function_for_context_retrieval, 
                    node_context_df, 
                    CONTEXT_VOLUME * 2,  # Double the context
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD * 0.8,  # Relax percentile
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,  # Lower threshold
                    edge_evidence, 
                    model_id=CHAT_MODEL_ID
                )
                
                # Use Mode 5's evaluation protocol
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "data sources", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms", "disease relationships"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options from the question",
                        "Step 2: For EACH option, search for direct evidence in retrieved_context",
                        "Step 3: Assign evidence strength: STRONG (explicit association), MEDIUM (indirect), WEAK (no mention)",
                        "Step 4: Select the option with STRONGEST evidence",
                        "Step 5: If no strong evidence, use biological principles and gene similarity"
                    ],
                    "critical_rule": "Evaluate ALL options. If retrieved_context is insufficient, acknowledge this and use best biological reasoning."
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Systematically evaluate EACH answer option. Prioritize options with explicit evidence in retrieved_context."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)


            elif MODE == "21":
                ### MODE 21: Separate Retrieval for Each Disease, Then Combine ###
                
                # Extract disease names from the question
                import re
                # Typical format: "associated with Disease1 and Disease2"
                diseases_match = re.search(r'associated with (.+?) and (.+?)\. Given', row["text"])
                
                if diseases_match:
                    disease1 = diseases_match.group(1).strip()
                    disease2 = diseases_match.group(2).strip()
                    
                    # Retrieve context for EACH disease separately
                    context1 = retrieve_context(
                        disease1, 
                        vectorstore, 
                        embedding_function_for_context_retrieval, 
                        node_context_df, 
                        CONTEXT_VOLUME,
                        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,
                        edge_evidence, 
                        model_id=CHAT_MODEL_ID
                    )
                    
                    context2 = retrieve_context(
                        disease2, 
                        vectorstore, 
                        embedding_function_for_context_retrieval, 
                        node_context_df, 
                        CONTEXT_VOLUME,
                        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,
                        edge_evidence, 
                        model_id=CHAT_MODEL_ID
                    )
                    
                    # Combine contexts with labels
                    combined_context = f"""
            [Context for {disease1}]:
            {context1}

            [Context for {disease2}]:
            {context2}
            """
                else:
                    # Fallback to standard retrieval
                    combined_context = retrieve_context(
                        row["text"], vectorstore, embedding_function_for_context_retrieval, 
                        node_context_df, CONTEXT_VOLUME * 2, 
                        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, 
                        model_id=CHAT_MODEL_ID
                    )
                
                reasoning_framework = {
                    "retrieved_context": combined_context,
                    "task": "Find the gene associated with BOTH diseases",
                    "strategy": "Look for genes that appear in BOTH disease contexts. This indicates shared genetic basis.",
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene-disease associations", "shared molecular mechanisms"]
                    },
                    "evaluation_protocol": [
                        "Step 1: List all answer options",
                        "Step 2: For EACH option, check if it appears in context for disease 1",
                        "Step 3: For EACH option, check if it appears in context for disease 2",
                        "Step 4: PRIORITIZE options that appear in BOTH contexts",
                        "Step 5: Select the option with evidence for both diseases"
                    ]
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Find the gene that connects BOTH diseases. Look for overlapping evidence."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "25":
                ### MODE 25: Temperature Ensemble Voting ###
                
                import re  # Add this import at the top
                from collections import Counter
                
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, 
                                          node_context_df, CONTEXT_VOLUME, 
                                          QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                          QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, 
                                          model_id=CHAT_MODEL_ID)
                
                reasoning_framework = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options",
                        "Step 2: For EACH option, find supporting evidence",
                        "Step 3: Score each option by evidence strength",
                        "Step 4: Select highest scoring option"
                    ]
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Evaluate ALL options systematically."""
                
                # Get 3 predictions with different temperatures
                answers = []
                for temp in [0.0, 0.3, 0.7]:
                    try:
                        output_temp = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=temp)
                        # Extract answer
                        match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_temp))
                        if match:
                            answers.append(match.group(1))
                    except Exception as e:
                        print(f"Warning: Failed to get response at temp={temp}: {e}")
                        continue
                
                # Majority vote
                if answers:
                    vote_counts = Counter(answers)
                    majority_answer = vote_counts.most_common(1)[0][0]
                    
                    # Format output with vote information
                    output = json.dumps({
                        "answer": majority_answer, 
                        "votes": dict(vote_counts), 
                        "note": f"Ensemble: {len(answers)}/3 temperatures succeeded"
                    })
                else:
                    # Fallback: if all temperatures failed, use single call with default temp
                    output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "26":
                ### MODE 26: Multi-Mode Ensemble Voting ###
                import re  # Add this import at the top
                from collections import Counter
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, 
                                          node_context_df, CONTEXT_VOLUME, 
                                          QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                          QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, 
                                          model_id=CHAT_MODEL_ID)
                
                answers = []
                
                # Run Mode 1 approach (JSON)
                context_json_1 = {"retrieved_information": context}
                json_str_1 = json.dumps(context_json_1, indent=2)
                prompt_1 = f"Context: {json_str_1}\n\nQuestion: {question}"
                output_1 = get_Gemini_response(prompt_1, SYSTEM_PROMPT, temperature=TEMPERATURE)
                match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_1))
                if match:
                    answers.append(('mode1', match.group(1)))
                
                # Run Mode 4 approach (Structured Reasoning)
                framework_4 = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                    },
                    "reasoning_steps": [
                        "1. Extract core question",
                        "2. Identify key entities",
                        "3. Filter context",
                        "4. Find evidence for each option",
                        "5. Select strongest molecular evidence"
                    ],
                    "quality_check": "Verify reasoning uses genetic/molecular evidence"
                }
                json_str_4 = json.dumps(framework_4, indent=2)
                prompt_4 = f"Biomedical Knowledge Context:\n{json_str_4}\n\nQuestion: {question}\n\nFollow reasoning_steps carefully."
                output_4 = get_Gemini_response(prompt_4, SYSTEM_PROMPT, temperature=TEMPERATURE)
                match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_4))
                if match:
                    answers.append(('mode4', match.group(1)))
                
                # Run Mode 5 approach (Option Evaluation)
                framework_5 = {
                    "retrieved_context": context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options",
                        "Step 2: For EACH option, find supporting evidence",
                        "Step 3: Score each option",
                        "Step 4: Select strongest evidence",
                        "Step 5: Double-check with similarity principle"
                    ],
                    "critical_rule": "Evaluate ALL options before deciding"
                }
                json_str_5 = json.dumps(framework_5, indent=2)
                prompt_5 = f"Biomedical Knowledge Context:\n{json_str_5}\n\nQuestion: {question}\n\nIMPORTANT: Evaluate EVERY answer option systematically."
                output_5 = get_Gemini_response(prompt_5, SYSTEM_PROMPT, temperature=TEMPERATURE)
                match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_5))
                if match:
                    answers.append(('mode5', match.group(1)))
                
                # Voting logic
                if len(answers) >= 2:
                    from collections import Counter
                    answer_only = [a[1] for a in answers]
                    vote_counts = Counter(answer_only)
                    
                    # If there's a clear majority (2 or 3 agree)
                    if vote_counts.most_common(1)[0][1] >= 2:
                        final_answer = vote_counts.most_common(1)[0][0]
                    else:
                        # All disagree - trust Mode 5 (best performer)
                        final_answer = answers[-1][1] if answers else None
                    
                    output = json.dumps({
                        "answer": final_answer,
                        "votes": {mode: ans for mode, ans in answers},
                        "agreement": f"{vote_counts.most_common(1)[0][1]}/3"
                    })
                else:
                    # Fallback to Mode 5 only
                    output = output_5

            elif MODE == "28":
                ### MODE 22: Explicit Empty Context Handling ###
                
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, 
                                          node_context_df, CONTEXT_VOLUME, 
                                          QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                          QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, 
                                          model_id=CHAT_MODEL_ID)
                
                # Check if context is actually useful
                context_is_empty = (not context or len(context.strip()) < 50)
                
                if context_is_empty:
                    # Special instructions for no-context scenarios
                    reasoning_framework = {
                        "situation": "No relevant context retrieved from knowledge graph",
                        "fallback_strategy": [
                            "1. This question requires inference from established biomedical knowledge",
                            "2. Focus on well-known gene-disease associations from literature",
                            "3. Consider gene families and functional similarities",
                            "4. For multi-disease questions: Look for genes with pleiotropic effects",
                            "5. Rare genes (ADGRV1, SLC14A2, EPDR1, etc.) may have unexpected associations"
                        ],
                        "evaluation_protocol": [
                            "Step 1: Identify what is being asked (gene connecting two diseases)",
                            "Step 2: Consider each option's known disease associations",
                            "Step 3: Prioritize options with documented multi-disease involvement",
                            "Step 4: Check for molecular mechanism plausibility",
                            "Step 5: Select based on strongest scientific rationale"
                        ],
                        "warning": "Without graph context, rely on established medical genetics knowledge and biological plausibility"
                    }
                else:
                    # Use Mode 5's approach when context exists
                    reasoning_framework = {
                        "retrieved_context": context,
                        "filtering_rules": {
                            "IGNORE": ["provenance", "symptoms"],
                            "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                        },
                        "evaluation_protocol": [
                            "Step 1: Extract all answer options",
                            "Step 2: For EACH option, find supporting evidence in context",
                            "Step 3: Score based on evidence strength",
                            "Step 4: Select strongest evidence match",
                            "Step 5: Verify with biological principles"
                        ],
                        "critical_rule": "Evaluate ALL options explicitly before deciding"
                    }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                enriched_prompt = f"""Biomedical Analysis Framework:
            {json_string_context}

            Question: {question}

            Analyze systematically and select the best answer."""
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            elif MODE == "30":
                ### MODE 30: ULTIMATE MULTI-STAGE PIPELINE ###
                print(f"\n[Question {index}] Starting Mode 30 pipeline...")
                output = mode_30_ultimate_pipeline(
                    row, question, vectorstore, embedding_function_for_context_retrieval,
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, CHAT_MODEL_ID,
                    get_Gemini_response, SYSTEM_PROMPT, TEMPERATURE, retrieve_context
                )


            elif MODE == "30a":
                ### MODE 30a: ABLATION - Multi-Retrieval ONLY (No Adaptive, No Ensemble) ###
                import re
                # Stage 1: Query Decomposition (same as Mode 30)
                diseases_match = re.search(r'associated with (.+?) and (.+?)\. Given', question)
                gene_list_match = re.search(r'Given list is:(.+)', question)
                
                diseases = []
                gene_options = []
                
                if diseases_match:
                    diseases = [diseases_match.group(1).strip(), diseases_match.group(2).strip()]
                if gene_list_match:
                    genes_text = gene_list_match.group(1)
                    gene_options = [g.strip() for g in genes_text.split(',') if g.strip()]
                
                # Stage 2: Multi-Strategy Retrieval (same as Mode 30)
                contexts = {}
                
                # Strategy 1: Full question
                try:
                    contexts['full'] = retrieve_context(
                        question, vectorstore, embedding_function_for_context_retrieval, 
                        node_context_df, CONTEXT_VOLUME, 
                        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, 
                        edge_evidence, model_id=CHAT_MODEL_ID
                    )
                except Exception as e:
                    contexts['full'] = ""
                
                # Strategy 2: Disease-specific
                if diseases:
                    for i, disease in enumerate(diseases):
                        try:
                            contexts[f'd{i}'] = retrieve_context(
                                disease, vectorstore, embedding_function_for_context_retrieval, 
                                node_context_df, CONTEXT_VOLUME // 2,
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,
                                edge_evidence, model_id=CHAT_MODEL_ID
                            )
                        except Exception as e:
                            contexts[f'd{i}'] = ""
                
                # Strategy 3: Gene-specific (top 3)
                if gene_options:
                    for i, gene in enumerate(gene_options[:3]):
                        try:
                            contexts[f'g{i}'] = retrieve_context(
                                f"{gene} disease association", 
                                vectorstore, embedding_function_for_context_retrieval, 
                                node_context_df, CONTEXT_VOLUME // 3,
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.6,
                                edge_evidence, model_id=CHAT_MODEL_ID
                            )
                        except Exception as e:
                            contexts[f'g{i}'] = ""
                
                # Strategy 4: Combined disease query
                if len(diseases) >= 2:
                    try:
                        combined_query = f"{diseases[0]} {diseases[1]} shared genetic basis"
                        contexts['combined'] = retrieve_context(
                            combined_query, vectorstore, embedding_function_for_context_retrieval, 
                            node_context_df, CONTEXT_VOLUME,
                            QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                            QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.75,
                            edge_evidence, model_id=CHAT_MODEL_ID
                        )
                    except Exception as e:
                        contexts['combined'] = ""
                
                # Simple aggregation (top 3 contexts)
                all_contexts = [v for v in contexts.values() if v and len(v) > 30]
                aggregated_context = "\n\n=== CONTEXT ===\n".join(all_contexts[:3]) if all_contexts else ""
                
                # FIXED reasoning (NO adaptation - same strategy for all questions)
                reasoning_framework = {
                    "retrieved_context": aggregated_context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options",
                        "Step 2: For EACH option, find supporting evidence",
                        "Step 3: Score each option based on evidence strength",
                        "Step 4: Select the option with strongest evidence"
                    ],
                    "critical_rule": "Evaluate ALL options systematically"
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            IMPORTANT: Evaluate EVERY answer option systematically."""
                
                # Single temperature (NO ensemble)
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)


            elif MODE == "30b":
                ### MODE 30b: ABLATION - Multi-Retrieval + Adaptive (No Ensemble) ###
                import re
                # Stage 1 & 2: Same as 30a
                diseases_match = re.search(r'associated with (.+?) and (.+?)\. Given', question)
                gene_list_match = re.search(r'Given list is:(.+)', question)
                
                diseases = []
                gene_options = []
                
                if diseases_match:
                    diseases = [diseases_match.group(1).strip(), diseases_match.group(2).strip()]
                if gene_list_match:
                    genes_text = gene_list_match.group(1)
                    gene_options = [g.strip() for g in genes_text.split(',') if g.strip()]
                
                contexts = {}
                
                try:
                    contexts['full'] = retrieve_context(
                        question, vectorstore, embedding_function_for_context_retrieval, 
                        node_context_df, CONTEXT_VOLUME, 
                        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, 
                        edge_evidence, model_id=CHAT_MODEL_ID
                    )
                except Exception as e:
                    contexts['full'] = ""
                
                if diseases:
                    for i, disease in enumerate(diseases):
                        try:
                            contexts[f'd{i}'] = retrieve_context(
                                disease, vectorstore, embedding_function_for_context_retrieval, 
                                node_context_df, CONTEXT_VOLUME // 2,
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,
                                edge_evidence, model_id=CHAT_MODEL_ID
                            )
                        except Exception as e:
                            contexts[f'd{i}'] = ""
                
                if gene_options:
                    for i, gene in enumerate(gene_options[:3]):
                        try:
                            contexts[f'g{i}'] = retrieve_context(
                                f"{gene} disease association", 
                                vectorstore, embedding_function_for_context_retrieval, 
                                node_context_df, CONTEXT_VOLUME // 3,
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.6,
                                edge_evidence, model_id=CHAT_MODEL_ID
                            )
                        except Exception as e:
                            contexts[f'g{i}'] = ""
                
                if len(diseases) >= 2:
                    try:
                        combined_query = f"{diseases[0]} {diseases[1]} shared genetic basis"
                        contexts['combined'] = retrieve_context(
                            combined_query, vectorstore, embedding_function_for_context_retrieval, 
                            node_context_df, CONTEXT_VOLUME,
                            QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                            QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.75,
                            edge_evidence, model_id=CHAT_MODEL_ID
                        )
                    except Exception as e:
                        contexts['combined'] = ""
                
                # Stage 3: Context Quality Assessment (KEY ADDITION)
                def assess_quality(ctx):
                    if not ctx or len(ctx.strip()) < 30:
                        return 0
                    score = len(ctx)
                    for gene in gene_options[:5]:
                        if gene.lower() in ctx.lower():
                            score += 100
                    for disease in diseases:
                        if disease.lower() in ctx.lower():
                            score += 50
                    return score
                
                context_scores = {k: assess_quality(v) for k, v in contexts.items()}
                sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
                top_contexts = [contexts[k] for k, score in sorted_contexts[:3] if score > 0]
                
                if top_contexts:
                    aggregated_context = "\n\n=== CONTEXT ===\n".join(top_contexts)
                    best_score = context_scores[sorted_contexts[0][0]]
                    context_quality = "HIGH" if best_score > 500 else ("MEDIUM" if best_score > 200 else "LOW")
                else:
                    aggregated_context = ""
                    context_quality = "NONE"
                
                # Stage 4: ADAPTIVE REASONING (KEY ADDITION - this is what 30a doesn't have)
                if context_quality in ["HIGH", "MEDIUM"]:
                    # Evidence-based reasoning
                    reasoning_framework = {
                        "context_quality": f"{context_quality} - Context available",
                        "retrieved_context": aggregated_context,
                        "filtering_rules": {
                            "IGNORE": ["provenance", "symptoms"],
                            "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                        },
                        "evaluation_protocol": [
                            "Step 1: Extract all gene options",
                            "Step 2: For EACH option, search context for evidence",
                            "Step 3: Score: 3 pts (both diseases), 2 pts (one + mechanism), 1 pt (weak), 0 pts (none)",
                            "Step 4: Select highest score"
                        ],
                        "critical_rule": "TRUST THE CONTEXT"
                    }
                else:  # LOW or NONE
                    # Knowledge-based reasoning
                    reasoning_framework = {
                        "context_quality": f"{context_quality} - Limited context",
                        "retrieved_context": aggregated_context,
                        "knowledge_strategy": [
                            "HLA genes → autoimmune diseases",
                            "TERT → cancers",
                            "Pleiotropic genes bridge diseases",
                            "Gene families share associations"
                        ],
                        "evaluation_protocol": [
                            "Step 1: List gene options",
                            "Step 2: Use biomedical knowledge for each",
                            "Step 3: Prioritize multi-disease genes",
                            "Step 4: Select most plausible"
                        ],
                        "warning": "LOW CONFIDENCE"
                    }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Evaluate all options based on context quality."""
                
                # Single temperature (NO ensemble)
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)


            elif MODE == "30c":
                ### MODE 30c: ABLATION - Multi-Retrieval + Ensemble (No Adaptive) ###
                import re
                from collections import Counter
                
                # Stage 1 & 2: Same multi-retrieval as 30a
                diseases_match = re.search(r'associated with (.+?) and (.+?)\. Given', question)
                gene_list_match = re.search(r'Given list is:(.+)', question)
                
                diseases = []
                gene_options = []
                
                if diseases_match:
                    diseases = [diseases_match.group(1).strip(), diseases_match.group(2).strip()]
                if gene_list_match:
                    genes_text = gene_list_match.group(1)
                    gene_options = [g.strip() for g in genes_text.split(',') if g.strip()]
                
                contexts = {}
                
                try:
                    contexts['full'] = retrieve_context(
                        question, vectorstore, embedding_function_for_context_retrieval, 
                        node_context_df, CONTEXT_VOLUME, 
                        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, 
                        edge_evidence, model_id=CHAT_MODEL_ID
                    )
                except Exception as e:
                    contexts['full'] = ""
                
                if diseases:
                    for i, disease in enumerate(diseases):
                        try:
                            contexts[f'd{i}'] = retrieve_context(
                                disease, vectorstore, embedding_function_for_context_retrieval, 
                                node_context_df, CONTEXT_VOLUME // 2,
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.7,
                                edge_evidence, model_id=CHAT_MODEL_ID
                            )
                        except Exception as e:
                            contexts[f'd{i}'] = ""
                
                if gene_options:
                    for i, gene in enumerate(gene_options[:3]):
                        try:
                            contexts[f'g{i}'] = retrieve_context(
                                f"{gene} disease association", 
                                vectorstore, embedding_function_for_context_retrieval, 
                                node_context_df, CONTEXT_VOLUME // 3,
                                QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.6,
                                edge_evidence, model_id=CHAT_MODEL_ID
                            )
                        except Exception as e:
                            contexts[f'g{i}'] = ""
                
                if len(diseases) >= 2:
                    try:
                        combined_query = f"{diseases[0]} {diseases[1]} shared genetic basis"
                        contexts['combined'] = retrieve_context(
                            combined_query, vectorstore, embedding_function_for_context_retrieval, 
                            node_context_df, CONTEXT_VOLUME,
                            QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                            QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY * 0.75,
                            edge_evidence, model_id=CHAT_MODEL_ID
                        )
                    except Exception as e:
                        contexts['combined'] = ""
                
                all_contexts = [v for v in contexts.values() if v and len(v) > 30]
                aggregated_context = "\n\n=== CONTEXT ===\n".join(all_contexts[:3]) if all_contexts else ""
                
                # FIXED reasoning (NO adaptation)
                reasoning_framework = {
                    "retrieved_context": aggregated_context,
                    "filtering_rules": {
                        "IGNORE": ["provenance", "symptoms"],
                        "PRIORITIZE": ["gene associations", "molecular mechanisms"]
                    },
                    "evaluation_protocol": [
                        "Step 1: Extract all answer options",
                        "Step 2: For EACH option, find supporting evidence",
                        "Step 3: Score each option",
                        "Step 4: Select strongest evidence"
                    ]
                }
                
                json_string_context = json.dumps(reasoning_framework, indent=2)
                enriched_prompt = f"""Biomedical Knowledge Context:
            {json_string_context}

            Question: {question}

            Evaluate all options systematically."""
                
                # ENSEMBLE (KEY ADDITION - this is what 30a doesn't have)
                answers = []
                for temp in [0.0, 0.3, 0.5]:
                    try:
                        output_temp = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=temp)
                        match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_temp))
                        if match:
                            candidate = match.group(1).strip()
                            if not gene_options or candidate in gene_options or any(candidate in opt for opt in gene_options):
                                answers.append(candidate)
                    except Exception as e:
                        continue
                
                # Voting
                if len(answers) >= 2:
                    vote_counts = Counter(answers)
                    if vote_counts.most_common(1)[0][1] >= 2:
                        final_answer = vote_counts.most_common(1)[0][0]
                    else:
                        final_answer = answers[0] if answers else None
                elif len(answers) == 1:
                    final_answer = answers[0]
                else:
                    # Fallback
                    output_fallback = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
                    match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(output_fallback))
                    final_answer = match.group(1) if match else "Error"
                
                output = json.dumps({"answer": final_answer, "ensemble_votes": len(answers)})

                

            answer_list.append((row["text"], row["correct_node"], output))
            
            # Save successful result immediately
            temp_df = pd.DataFrame([(row["text"], row["correct_node"], output)], columns=["question", "correct_answer", "llm_answer"])
            temp_df.to_csv(output_file, mode="a", index=False, header=False)
            # No rate limit delay needed

        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))
            
            # Save error result immediately
            temp_df = pd.DataFrame([(row["text"], row["correct_node"], "Error")], columns=["question", "correct_answer", "llm_answer"])
            temp_df.to_csv(output_file, mode="a", index=False, header=False)
            # No rate limit delay needed

    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))
        
        
if __name__ == "__main__":
    main()
