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


SCOPE_EXPLORATION_PROMPT = """You are analyzing a codebase to find all files relevant to a specific question.

SCOPE QUESTION:
{scope_question}

FILE TREE:
{file_tree}

CANDIDATE FILES (already found by keyword search):
{candidate_files}

CANDIDATE FILE CONTENTS:
{candidate_contents}

Your task:
1. Review the candidate files and their contents
2. Identify what additional files should be examined (imports, related tests, configs)
3. Determine if you have enough context to answer the scope question
4. Note important line numbers/ranges for key code locations

When examining files, track important line numbers for:
- Key function or class definitions relevant to the question
- Important logic, algorithms, or configuration
- Entry points and initialization code
- Error handling and edge cases

Respond with JSON:
{{
    "additional_files_needed": ["path/to/file1.py", "path/to/file2.py"],
    "reasoning": "Why these files are needed",
    "sufficient_context": true/false,
    "preliminary_insights": "What you've learned so far. Include notable file:line references like 'the main logic is in src/auth.py:45-78'",
    "key_locations": [
        {{"path": "path/to/file.py", "line_start": 45, "line_end": 78, "description": "Main authentication flow"}}
    ]
}}

If sufficient_context is true, additional_files_needed should be empty.
The key_locations array should contain the most important code locations discovered so far.
"""


SCOPE_GENERATION_PROMPT = """Generate a focused context document for the following scope question.

SCOPE QUESTION:
{scope_question}

RELEVANT FILES AND CONTENTS:
{relevant_files}

ANALYSIS INSIGHTS:
{insights}

Create a markdown document with:
1. Summary - Direct answer to the scope question
2. Relevant sections based on what's important for this specific topic
   (could be API endpoints, data models, processing logic, etc. - use your judgment)
3. Key Files - List of files the reader should examine with specific line references
4. Usage Examples / Related Tests - If available, show how this functionality is used

When referencing code locations, use the format `path/to/file.py:line` or `path/to/file.py:start-end`.

Be concise but thorough. Focus only on information relevant to the scope question.
Do NOT include a generic structure - tailor sections to what matters for this topic.

IMPORTANT: In the "Key Files" section, include specific line numbers where relevant code is located.
Format references as: `path/to/file.py:45-78` - Brief description of what this code does.
"""
