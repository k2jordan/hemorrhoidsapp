"""
Minimal Test Runner - Bypasses the issue
This is a stripped-down version that should work
"""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("""
╔═══════════════════════════════════════════════════════════╗
║    MEDICAL CHATBOT RIGOROUS TESTING FRAMEWORK              ║
║    Testing: Hemorrhoid & Constipation Management App       ║
╚═══════════════════════════════════════════════════════════╝
""")

print("DEBUG: About to import patient_chatbot...")
try:
    from patient_chatbot import load_vectorstore, PatientChatbot
    print("✓ Imported patient_chatbot")
except Exception as e:
    print(f"✗ Failed to import patient_chatbot: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("DEBUG: About to import testing_framework...")
try:
    from testing_framework import LLMJudgeEvaluator, HumanEvaluationInterface
    print("✓ Imported testing_framework")
except Exception as e:
    print(f"✗ Failed to import testing_framework: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("DEBUG: About to load vectorstore...")
try:
    vectorstore = load_vectorstore("./faiss_index")
    print("✓ Loaded vectorstore")
except Exception as e:
    print(f"✗ Failed to load vectorstore: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("DEBUG: About to create chatbot...")
try:
    chatbot = PatientChatbot(vectorstore, "test_patient")
    print("✓ Created chatbot")
except Exception as e:
    print(f"✗ Failed to create chatbot: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("DEBUG: About to load manual test cases...")
try:
    manual_path = Path("test_data/manual_test_cases.json")
    if manual_path.exists():
        with open(manual_path, 'r') as f:
            data = json.load(f)
        test_cases = [
            {
                'id': q['id'],
                'category': q['category'],
                'question': q['title'],
                'metadata': {
                    'source': q['source'],
                    'url': q.get('url', ''),
                    'body': q.get('body', '')
                }
            }
            for q in data['questions']
        ]
        print(f"✓ Loaded {len(test_cases)} manual test cases")
    else:
        print(f"✗ Manual test cases not found at {manual_path}")
        exit(1)
except Exception as e:
    print(f"✗ Failed to load test cases: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✅ ALL INITIALIZATION SUCCESSFUL!")
print("="*60)

print("\nTest Configuration:")
print("1. Generate responses only (save for later)")
print("2. Generate responses + LLM evaluation")
print("3. Generate responses + Human evaluation")
print("4. Generate responses + BOTH (LLM and Human)")

choice = input("\nSelect option (1-4): ").strip()

# Generate responses
print("\n" + "="*60)
print(f"GENERATING RESPONSES FOR {len(test_cases)} TEST CASES")
print("="*60)

responses = []
for i, test_case in enumerate(test_cases, 1):
    print(f"\n[{i}/{len(test_cases)}] Processing: {test_case['question'][:60]}...")
    
    try:
        response = chatbot.chat(test_case['question'])
        responses.append({
            'test_case': test_case,
            'response': response
        })
        print(f"  ✓ Response generated ({len(response)} chars)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        responses.append({
            'test_case': test_case,
            'response': None,
            'error': str(e)
        })

# Save responses
output_dir = Path("test_results")
output_dir.mkdir(exist_ok=True)

responses_file = output_dir / "generated_responses.json"
with open(responses_file, 'w') as f:
    json.dump({
        'total_cases': len(responses),
        'successful': sum(1 for r in responses if r['response'] is not None),
        'failed': sum(1 for r in responses if r['response'] is None),
        'results': responses
    }, f, indent=2)

print(f"\n✓ Responses saved to {responses_file}")

# Run evaluations based on choice
if choice in ['2', '4']:
    print("\n" + "="*60)
    print("RUNNING LLM-AS-JUDGE EVALUATION")
    print("="*60)
    
    evaluator = LLMJudgeEvaluator()
    
    valid_responses = [r for r in responses if r['response'] is not None]
    test_cases_valid = [r['test_case'] for r in valid_responses]
    responses_valid = [r['response'] for r in valid_responses]
    
    evaluation_results = evaluator.batch_evaluate(test_cases_valid, responses_valid)
    evaluator.save_evaluation_results(evaluation_results)

if choice in ['3', '4']:
    print("\n" + "="*60)
    print("RUNNING HUMAN EVALUATION")
    print("="*60)
    
    human_eval = HumanEvaluationInterface()
    
    valid_responses = [r for r in responses if r['response'] is not None]
    test_cases_valid = [r['test_case'] for r in valid_responses]
    responses_valid = [r['response'] for r in valid_responses]
    
    num_to_review = int(input(f"\nHow many cases to review (max {len(valid_responses)})? "))
    num_to_review = min(num_to_review, len(valid_responses))
    
    human_results = human_eval.batch_evaluate(
        test_cases_valid[:num_to_review],
        responses_valid[:num_to_review]
    )
    human_eval.save_human_evaluations(human_results)

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print("\nResults saved to ./test_results/")

if choice == '4':
    print("\nNow run: python compare_evaluations.py")