import os
import json
from datetime import datetime

class FindingsAnalyzer:
    def __init__(self):
        self.findings_dir = "analysis_findings"
        if not os.path.exists(self.findings_dir):
            os.makedirs(self.findings_dir)
    
    def analyze_comparison(self, comparison_file):
        """Analyze comparison results and generate detailed findings"""
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        comparison = data['comparison']
        user_name = data['user_name']
        
        # Sort features by score
        feature_scores = comparison['feature_scores']
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Generate findings
        findings = {
            'player_name': user_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'overall_score': comparison['overall_score'],
            'analysis': {
                'strengths': [],
                'improvements': [],
                'key_differences': [],
                'recommendations': []
            }
        }
        
        # Analyze strengths (top 3 scores)
        for feature, data in sorted_features[:3]:
            findings['analysis']['strengths'].append({
                'feature': feature,
                'score': data['score'],
                'details': f"{feature}: {data['score']:.1f}/100 - "
                          f"{'Very close to' if data['score'] > 80 else 'Similar to'} "
                          f"Curry's form (difference: {abs(data['difference']):.3f})"
            })
        
        # Analyze areas for improvement (bottom 3 scores)
        for feature, data in sorted_features[-3:]:
            findings['analysis']['improvements'].append({
                'feature': feature,
                'score': data['score'],
                'details': f"{feature}: {data['score']:.1f}/100 - "
                          f"{'Significant' if data['score'] < 60 else 'Moderate'} "
                          f"difference from Curry's form"
            })
        
        # Analyze key differences
        for feature, data in feature_scores.items():
            if abs(data['difference']) > data['reference_std']:
                direction = "higher" if data['difference'] > 0 else "lower"
                findings['analysis']['key_differences'].append({
                    'feature': feature,
                    'difference': data['difference'],
                    'details': f"{feature}: {abs(data['difference']):.3f} units {direction} "
                              f"than Curry's average of {data['reference_mean']:.3f}"
                })
        
        # Generate recommendations
        self._generate_recommendations(findings, feature_scores)
        
        return findings
    
    def _generate_recommendations(self, findings, feature_scores):
        """Generate specific recommendations based on the analysis"""
        recommendations = []
        
        # Analyze elbow and shoulder angles
        if 'Elbow Angle' in feature_scores:
            elbow_data = feature_scores['Elbow Angle']
            if abs(elbow_data['difference']) > 20:
                direction = "decrease" if elbow_data['difference'] > 0 else "increase"
                recommendations.append(
                    f"Work on {direction}ing elbow angle during release - "
                    f"current: {elbow_data['user_value']:.1f}°, "
                    f"target: {elbow_data['reference_mean']:.1f}°"
                )
        
        # Analyze release point
        if all(k in feature_scores for k in ['Release X', 'Release Y']):
            release_x = feature_scores['Release X']
            release_y = feature_scores['Release Y']
            if release_y['score'] < 70:
                recommendations.append(
                    f"Adjust release height - "
                    f"current: {release_y['user_value']:.3f}, "
                    f"target: {release_y['reference_mean']:.3f}"
                )
        
        # Analyze shot arc
        if 'Shot Arc' in feature_scores:
            arc_data = feature_scores['Shot Arc']
            if arc_data['score'] < 70:
                recommendations.append(
                    f"Adjust shot arc - "
                    f"current: {arc_data['user_value']:.3f} rad, "
                    f"target: {arc_data['reference_mean']:.3f} rad"
                )
        
        findings['analysis']['recommendations'] = recommendations
    
    def save_findings(self, findings, output_format='both'):
        """Save findings in specified format (txt, json, or both)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        player_name = findings['player_name']
        base_path = os.path.join(self.findings_dir, f"{player_name}_{timestamp}")
        
        if output_format in ['json', 'both']:
            # Save JSON format
            json_path = f"{base_path}_findings.json"
            with open(json_path, 'w') as f:
                json.dump(findings, f, indent=4)
        
        if output_format in ['txt', 'both']:
            # Save human-readable format
            txt_path = f"{base_path}_findings.txt"
            with open(txt_path, 'w') as f:
                f.write(f"Shot Form Analysis for {player_name}\n")
                f.write(f"Date: {findings['timestamp']}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Overall Score: {findings['overall_score']:.1f}/100\n\n")
                
                f.write("STRENGTHS:\n")
                f.write("-" * 20 + "\n")
                for strength in findings['analysis']['strengths']:
                    f.write(f"• {strength['details']}\n")
                f.write("\n")
                
                f.write("AREAS FOR IMPROVEMENT:\n")
                f.write("-" * 20 + "\n")
                for improvement in findings['analysis']['improvements']:
                    f.write(f"• {improvement['details']}\n")
                f.write("\n")
                
                f.write("KEY DIFFERENCES FROM CURRY'S FORM:\n")
                f.write("-" * 20 + "\n")
                for diff in findings['analysis']['key_differences']:
                    f.write(f"• {diff['details']}\n")
                f.write("\n")
                
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 20 + "\n")
                for rec in findings['analysis']['recommendations']:
                    f.write(f"• {rec}\n")
        
        return base_path

def main():
    # Initialize analyzer
    analyzer = FindingsAnalyzer()
    
    # Find most recent comparison file
    comparison_dir = "comparison_results"
    comparison_files = [f for f in os.listdir(comparison_dir) 
                       if f.endswith('_results.json')]
    if not comparison_files:
        print("No comparison results found!")
        return
    
    # Get most recent file
    latest_file = max(comparison_files, 
                     key=lambda x: os.path.getctime(os.path.join(comparison_dir, x)))
    comparison_path = os.path.join(comparison_dir, latest_file)
    
    print(f"Analyzing comparison results from: {latest_file}")
    
    # Analyze findings
    findings = analyzer.analyze_comparison(comparison_path)
    
    # Save findings in both formats
    base_path = analyzer.save_findings(findings)
    
    print("\nAnalysis complete!")
    print(f"Findings saved to:")
    print(f"- {base_path}_findings.txt (human-readable)")
    print(f"- {base_path}_findings.json (machine-readable)")

if __name__ == "__main__":
    main() 