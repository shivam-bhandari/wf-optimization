#!/usr/bin/env python3
"""
Code Review Analysis Script

This script analyzes the codebase and generates metrics useful for code review.
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json


class CodeAnalyzer:
    """Analyze Python code files for code review metrics."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir / "src"
        self.stats = defaultdict(lambda: defaultdict(int))
        self.issues = []
        self.files_analyzed = []
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            rel_path = str(file_path.relative_to(self.root_dir))
            self.files_analyzed.append(rel_path)
            
            file_stats = {
                'file': rel_path,
                'lines': len(content.splitlines()),
                'functions': 0,
                'classes': 0,
                'imports': 0,
                'complexity': 0,
                'has_docstring': False,
                'issues': []
            }
            
            # Walk through AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    file_stats['functions'] += 1
                    # Check for docstrings
                    if ast.get_docstring(node):
                        file_stats['has_docstring'] = True
                    
                    # Simple complexity: count branches
                    complexity = self._calculate_complexity(node)
                    file_stats['complexity'] = max(file_stats['complexity'], complexity)
                    
                    # Check for long functions
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if func_lines > 50:
                        file_stats['issues'].append(
                            f"Function '{node.name}' is long ({func_lines} lines), consider refactoring"
                        )
                
                elif isinstance(node, ast.ClassDef):
                    file_stats['classes'] += 1
                    if ast.get_docstring(node):
                        file_stats['has_docstring'] = True
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    file_stats['imports'] += 1
                    # Check for wildcard imports
                    if isinstance(node, ast.ImportFrom) and node.names:
                        if any(name.name == '*' for name in node.names):
                            file_stats['issues'].append(
                                f"Wildcard import found at line {node.lineno}: {ast.get_source_segment(content, node)}"
                            )
            
            # Check for module docstring
            if ast.get_docstring(tree):
                file_stats['has_docstring'] = True
            
            # File size check
            if file_stats['lines'] > 500:
                file_stats['issues'].append(
                    f"Large file ({file_stats['lines']} lines), consider splitting into smaller modules"
                )
            
            if file_stats['issues']:
                self.issues.extend([(rel_path, issue) for issue in file_stats['issues']])
            
            return file_stats
            
        except SyntaxError as e:
            return {
                'file': str(file_path.relative_to(self.root_dir)),
                'error': f"Syntax error: {e}",
                'issues': [f"Syntax error at line {e.lineno}: {e.msg}"]
            }
        except Exception as e:
            return {
                'file': str(file_path.relative_to(self.root_dir)),
                'error': str(e)
            }
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Simple cyclomatic complexity calculation."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def analyze_directory(self, directory: Path) -> List[Dict]:
        """Analyze all Python files in a directory."""
        results = []
        
        if not directory.exists():
            return results
        
        for py_file in directory.rglob("*.py"):
            # Skip __pycache__
            if "__pycache__" in str(py_file):
                continue
            
            file_stats = self.analyze_file(py_file)
            results.append(file_stats)
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        # Analyze src directory
        src_results = self.analyze_directory(self.src_dir)
        
        # Calculate totals
        total_lines = sum(r.get('lines', 0) for r in src_results)
        total_functions = sum(r.get('functions', 0) for r in src_results)
        total_classes = sum(r.get('classes', 0) for r in src_results)
        
        # Find largest files
        large_files = sorted(
            [r for r in src_results if r.get('lines', 0) > 200],
            key=lambda x: x.get('lines', 0),
            reverse=True
        )[:10]
        
        # Find files without docstrings
        files_without_docstrings = [
            r['file'] for r in src_results 
            if not r.get('has_docstring', False) and r.get('lines', 0) > 20
        ]
        
        # Files with issues
        files_with_issues = {
            file: [issue for f, issue in self.issues if f == file]
            for file in set(f for f, _ in self.issues)
        }
        
        report = {
            'summary': {
                'files_analyzed': len(self.files_analyzed),
                'total_lines': total_lines,
                'total_functions': total_functions,
                'total_classes': total_classes,
                'avg_lines_per_file': total_lines / len(src_results) if src_results else 0,
                'avg_functions_per_file': total_functions / len(src_results) if src_results else 0,
                'issues_found': len(self.issues)
            },
            'largest_files': [
                {
                    'file': r['file'],
                    'lines': r.get('lines', 0),
                    'functions': r.get('functions', 0),
                    'classes': r.get('classes', 0)
                }
                for r in large_files
            ],
            'files_without_docstrings': files_without_docstrings,
            'issues_by_file': files_with_issues,
            'all_issues': [{'file': f, 'issue': issue} for f, issue in self.issues]
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted report to console."""
        print("=" * 80)
        print("CODE REVIEW ANALYSIS REPORT")
        print("=" * 80)
        print()
        
        summary = report['summary']
        print("SUMMARY")
        print("-" * 80)
        print(f"Files Analyzed:        {summary['files_analyzed']}")
        print(f"Total Lines of Code:   {summary['total_lines']:,}")
        print(f"Total Functions:       {summary['total_functions']}")
        print(f"Total Classes:         {summary['total_classes']}")
        print(f"Avg Lines/File:        {summary['avg_lines_per_file']:.1f}")
        print(f"Avg Functions/File:    {summary['avg_functions_per_file']:.1f}")
        print(f"Issues Found:          {summary['issues_found']}")
        print()
        
        if report['largest_files']:
            print("LARGEST FILES (>200 lines)")
            print("-" * 80)
            for item in report['largest_files']:
                print(f"{item['file']:50} {item['lines']:5} lines "
                      f"({item['functions']} funcs, {item['classes']} classes)")
            print()
        
        if report['files_without_docstrings']:
            print("FILES WITHOUT DOCSTRINGS (>20 lines)")
            print("-" * 80)
            for file in report['files_without_docstrings']:
                print(f"  {file}")
            print()
        
        if report['all_issues']:
            print("ISSUES FOUND")
            print("-" * 80)
            current_file = None
            for item in report['all_issues']:
                if item['file'] != current_file:
                    current_file = item['file']
                    print(f"\n{current_file}:")
                print(f"  ⚠  {item['issue']}")
            print()
        
        print("=" * 80)
        print("Review complete. See detailed report JSON file for more information.")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze codebase for code review")
    parser.add_argument(
        '--dir',
        type=str,
        default='.',
        help='Root directory of the project (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only output JSON, no console output'
    )
    
    args = parser.parse_args()
    
    analyzer = CodeAnalyzer(args.dir)
    report = analyzer.generate_report()
    
    if not args.json_only:
        analyzer.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        if not args.json_only:
            print(f"\nReport saved to: {args.output}")
    elif args.json_only:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

