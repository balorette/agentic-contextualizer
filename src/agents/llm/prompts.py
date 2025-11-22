"""Prompt templates for LLM calls."""

CODE_ANALYSIS_PROMPT = """You are analyzing a codebase to extract architectural insights.

PROJECT STRUCTURE:
{file_tree}

KEY FILES CONTENT:
{key_files_content}

USER SUMMARY:
{user_summary}

Analyze this codebase and provide:
1. Architectural patterns (e.g., MVC, microservices, monolith)
2. Technology stack
3. Coding conventions (error handling, testing patterns, state management)
4. Notable design decisions

Format your response as structured JSON with keys: architecture_patterns, tech_stack, coding_conventions, insights.
"""


CONTEXT_GENERATION_PROMPT = """Generate a comprehensive context file for this codebase.

PROJECT METADATA:
{project_metadata}

CODE ANALYSIS:
{code_analysis}

USER SUMMARY:
{user_summary}

Create a markdown document following this structure:
1. Architecture Overview
2. Key Commands (build, test, run, deploy)
3. Code Patterns
4. Entry Points

Be concise but thorough. Focus on what an AI agent needs to understand the codebase quickly.
"""


REFINEMENT_PROMPT = """Update the existing context file based on user feedback.

CURRENT CONTEXT:
{current_context}

USER REQUEST:
{user_request}

Generate the updated context file incorporating the requested changes.
"""
