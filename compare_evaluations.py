"""
LLM vs Human Evaluation Comparison
Analyzes agreement and disagreement between LLM-as-judge and human evaluators
"""

import json
from pathlib import Path
from typing import Dict, List
import statistics

class EvaluationComparison:
    """Compare LLM and Human evaluations on the same test cases"""
    
    def __init__(self, 
                 llm_results_file: str = "test_results/evaluation_results.json",
                 human_results_file: str = "test_results/human_evaluations.json"):
        
        self.llm_file = Path(llm_results_file)
        self.human_file = Path(human_results_file)
        
        if not self.llm_file.exists():
            raise FileNotFoundError(f"LLM results not found: {llm_results_file}")
        
        if not self.human_file.exists():
            raise FileNotFoundError(f"Human results not found: {human_results_file}")
        
        # Load both sets of results
        with open(self.llm_file, 'r') as f:
            self.llm_data = json.load(f)
        
        with open(self.human_file, 'r') as f:
            self.human_data = json.load(f)
        
        print(f"✓ Loaded {len(self.llm_data['detailed_results'])} LLM evaluations")
        print(f"✓ Loaded {len(self.human_data['results'])} human evaluations")
    
    def match_evaluations(self) -> List[Dict]:
        """Match LLM and human evaluations by test case ID"""
        
        # Create lookup by test case ID
        llm_by_id = {
            result['test_case_id']: result 
            for result in self.llm_data['detailed_results']
        }
        
        human_by_id = {
            result['test_case_id']: result 
            for result in self.human_data['results']
        }
        
        # Find matches
        matched = []
        for test_id in llm_by_id.keys():
            if test_id in human_by_id:
                matched.append({
                    'test_case_id': test_id,
                    'question': llm_by_id[test_id]['question'],
                    'llm': llm_by_id[test_id],
                    'human': human_by_id[test_id]
                })
        
        print(f"✓ Matched {len(matched)} cases evaluated by both LLM and human")
        return matched
    
    def calculate_agreement(self, matched_cases: List[Dict]) -> Dict:
        """Calculate agreement metrics between LLM and human"""
        
        agreements = {
            'verdict': 0,  # PASS/REVISE/FAIL
            'overall_similar': 0  # Within 10% on overall score
        }
        
        disagreements = []
        score_differences = []
        
        for case in matched_cases:
            llm_eval = case['llm']['evaluation']
            human_eval = case['human']['evaluation']
            
            # Get LLM verdict
            llm_verdict = llm_eval.get('recommended_action', 'UNKNOWN')
            
            # Get human verdict
            human_verdict = human_eval.get('verdict', 'UNKNOWN')
            
            # Check verdict agreement
            if llm_verdict == human_verdict:
                agreements['verdict'] += 1
            else:
                disagreements.append({
                    'test_case_id': case['test_case_id'],
                    'question': case['question'][:80] + '...',
                    'llm_verdict': llm_verdict,
                    'human_verdict': human_verdict
                })
            
            # Compare scores (convert to same scale)
            llm_score = llm_eval.get('overall_assessment', {}).get('percentage', 0)
            human_score = human_eval.get('overall_rating', 0) * 20  # Convert 1-5 to 0-100
            
            score_diff = abs(llm_score - human_score)
            score_differences.append(score_diff)
            
            if score_diff <= 10:  # Within 10%
                agreements['overall_similar'] += 1
        
        total = len(matched_cases)
        
        return {
            'total_cases': total,
            'verdict_agreement': {
                'count': agreements['verdict'],
                'percentage': (agreements['verdict'] / total * 100) if total > 0 else 0
            },
            'score_similarity': {
                'within_10_percent': agreements['overall_similar'],
                'percentage': (agreements['overall_similar'] / total * 100) if total > 0 else 0
            },
            'avg_score_difference': statistics.mean(score_differences) if score_differences else 0,
            'disagreements': disagreements,
            'score_differences': score_differences
        }
    
    def find_major_disagreements(self, matched_cases: List[Dict]) -> List[Dict]:
        """Find cases where LLM and human strongly disagreed"""
        
        major_disagreements = []
        
        for case in matched_cases:
            llm_eval = case['llm']['evaluation']
            human_eval = case['human']['evaluation']
            
            llm_verdict = llm_eval.get('recommended_action', 'UNKNOWN')
            human_verdict = human_eval.get('verdict', 'UNKNOWN')
            
            # Major disagreement: one says PASS, other says FAIL
            if (llm_verdict == 'PASS' and human_verdict == 'FAIL') or \
               (llm_verdict == 'FAIL' and human_verdict == 'PASS'):
                
                llm_score = llm_eval.get('overall_assessment', {}).get('percentage', 0)
                human_score = human_eval.get('overall_rating', 0) * 20
                
                major_disagreements.append({
                    'test_case_id': case['test_case_id'],
                    'question': case['question'],
                    'response': case['llm']['response'][:200] + '...',
                    'llm_verdict': llm_verdict,
                    'llm_score': f"{llm_score:.1f}%",
                    'human_verdict': human_verdict,
                    'human_score': f"{human_score:.1f}%",
                    'human_comments': human_eval.get('comments', '')
                })
        
        return major_disagreements
    
    def analyze_dimension_agreement(self, matched_cases: List[Dict]) -> Dict:
        """Compare agreement on specific evaluation dimensions"""
        
        # Map LLM dimensions to human dimensions
        dimension_mapping = {
            'medical_accuracy': 'medical_accuracy',
            'safety': 'medical_accuracy',  # Both relate to correctness
            'patient_friendliness': 'empathy',
            'actionability': 'actionability',
            'scope_appropriateness': 'appropriateness'
        }
        
        dimension_diffs = {dim: [] for dim in dimension_mapping.values()}
        
        for case in matched_cases:
            llm_eval = case['llm']['evaluation']
            human_eval = case['human']['evaluation']
            
            human_ratings = human_eval.get('ratings', {})
            
            for llm_dim, human_dim in dimension_mapping.items():
                if llm_dim in llm_eval and human_dim in human_ratings:
                    llm_score = llm_eval[llm_dim].get('score', 0)  # 0-10
                    human_score = human_ratings[human_dim]  # 1-5
                    
                    # Normalize both to 0-10 scale
                    llm_normalized = llm_score
                    human_normalized = (human_score - 1) * 2.5  # Convert 1-5 to 0-10
                    
                    diff = abs(llm_normalized - human_normalized)
                    dimension_diffs[human_dim].append(diff)
        
        # Calculate average differences
        avg_diffs = {}
        for dim, diffs in dimension_diffs.items():
            if diffs:
                avg_diffs[dim] = {
                    'avg_difference': statistics.mean(diffs),
                    'max_difference': max(diffs),
                    'samples': len(diffs)
                }
        
        return avg_diffs
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        
        print("\n" + "="*80)
        print("LLM vs HUMAN EVALUATION COMPARISON REPORT")
        print("="*80)
        
        # Match cases
        matched = self.match_evaluations()
        
        if not matched:
            print("\n⚠️  No matching cases found between LLM and human evaluations!")
            print("Make sure both evaluations were run on the same test cases.")
            return
        
        # Overall agreement
        print("\n" + "="*80)
        print("OVERALL AGREEMENT")
        print("="*80)
        
        agreement = self.calculate_agreement(matched)
        
        print(f"\n📊 Verdict Agreement:")
        print(f"   {agreement['verdict_agreement']['count']}/{agreement['total_cases']} cases")
        print(f"   {agreement['verdict_agreement']['percentage']:.1f}% agreement on PASS/REVISE/FAIL")
        
        print(f"\n📊 Score Similarity:")
        print(f"   {agreement['score_similarity']['within_10_percent']}/{agreement['total_cases']} cases within 10%")
        print(f"   {agreement['score_similarity']['percentage']:.1f}% score similarity")
        print(f"   Average score difference: {agreement['avg_score_difference']:.1f}%")
        
        # Disagreements
        if agreement['disagreements']:
            print(f"\n⚠️  VERDICT DISAGREEMENTS ({len(agreement['disagreements'])} cases):")
            for i, disagree in enumerate(agreement['disagreements'][:5], 1):
                print(f"\n   {i}. {disagree['test_case_id']}")
                print(f"      Question: {disagree['question']}")
                print(f"      LLM: {disagree['llm_verdict']} | Human: {disagree['human_verdict']}")
            
            if len(agreement['disagreements']) > 5:
                print(f"\n      ... and {len(agreement['disagreements']) - 5} more")
        
        # Major disagreements
        print("\n" + "="*80)
        print("MAJOR DISAGREEMENTS (PASS vs FAIL)")
        print("="*80)
        
        major = self.find_major_disagreements(matched)
        
        if major:
            print(f"\n⚠️  Found {len(major)} cases with major disagreement:\n")
            
            for i, case in enumerate(major, 1):
                print(f"{i}. {case['test_case_id']}")
                print(f"   Question: {case['question'][:100]}...")
                print(f"   LLM: {case['llm_verdict']} ({case['llm_score']})")
                print(f"   Human: {case['human_verdict']} ({case['human_score']})")
                if case['human_comments']:
                    print(f"   Human comment: {case['human_comments']}")
                print()
        else:
            print("\n✅ No major disagreements found!")
        
        # Dimension analysis
        print("="*80)
        print("DIMENSION-BY-DIMENSION AGREEMENT")
        print("="*80)
        
        dim_analysis = self.analyze_dimension_agreement(matched)
        
        print("\nAverage differences by dimension (0-10 scale):")
        for dim, stats in sorted(dim_analysis.items(), key=lambda x: x[1]['avg_difference']):
            emoji = "✓" if stats['avg_difference'] < 2 else "⚠️" if stats['avg_difference'] < 3 else "✗"
            print(f"   {emoji} {dim}: {stats['avg_difference']:.2f} avg diff (max: {stats['max_difference']:.2f})")
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        verdict_agreement_pct = agreement['verdict_agreement']['percentage']
        
        if verdict_agreement_pct >= 80:
            print("\n✅ STRONG AGREEMENT (≥80%)")
            print("   LLM evaluations are closely aligned with human judgment.")
            print("   The LLM-as-judge can be trusted for automated testing.")
        elif verdict_agreement_pct >= 60:
            print("\n⚠️  MODERATE AGREEMENT (60-79%)")
            print("   LLM evaluations are somewhat aligned with human judgment.")
            print("   Review disagreement cases to understand differences.")
            print("   Consider:")
            print("   - Adjusting LLM evaluation prompt for clarity")
            print("   - Adding more specific evaluation criteria")
        else:
            print("\n❌ LOW AGREEMENT (<60%)")
            print("   Significant disagreement between LLM and human evaluations.")
            print("   Action needed:")
            print("   - Review major disagreement cases carefully")
            print("   - Revise LLM evaluation prompt and criteria")
            print("   - Consider if human evaluators need clearer guidelines")
        
        print("\n" + "="*80)
        
        # Save detailed comparison
        self.save_comparison(matched, agreement, major, dim_analysis)
    
    def save_comparison(self, matched, agreement, major_disagreements, dimension_analysis):
        """Save detailed comparison to file"""
        
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / "llm_vs_human_comparison.json"
        
        comparison_data = {
            'total_matched_cases': len(matched),
            'agreement_metrics': agreement,
            'major_disagreements': major_disagreements,
            'dimension_analysis': dimension_analysis,
            'matched_cases': [
                {
                    'test_case_id': case['test_case_id'],
                    'question': case['question'],
                    'llm_verdict': case['llm']['evaluation'].get('recommended_action'),
                    'llm_score': case['llm']['evaluation'].get('overall_assessment', {}).get('percentage'),
                    'human_verdict': case['human']['evaluation'].get('verdict'),
                    'human_score': case['human']['evaluation'].get('overall_rating', 0) * 20
                }
                for case in matched
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n✓ Detailed comparison saved to {filepath}")

def main():
    """Run the comparison"""
    try:
        comparator = EvaluationComparison()
        comparator.generate_report()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run both LLM and human evaluations first!")
        print("Run: python test_runner.py")
        print("Choose option 5: Manual test cases + BOTH LLM and Human")

if __name__ == "__main__":
    main()