"""Integration test: progressive disclosure tool flow."""

import pytest
from src.agents.tools.backends import InMemoryFileBackend
from src.agents.backends import ASTFileAnalysisBackend
from src.agents.file_access import SmartFileAccess
from src.agents.tools.progressive import create_progressive_tools


PYTHON_AUTH = '''"""Authentication module."""

import jwt
from datetime import datetime, timedelta
from .models import User, Session

SECRET_KEY = "changeme"

def create_token(user: User) -> str:
    """Create a JWT token for the user."""
    payload = {
        "user_id": user.id,
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str) -> dict | None:
    """Verify and decode a JWT token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None

class AuthMiddleware:
    """Middleware for request authentication."""

    def __init__(self, secret: str = SECRET_KEY):
        self.secret = secret

    def authenticate(self, request) -> User | None:
        """Extract and verify token from request."""
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        payload = verify_token(token)
        if payload:
            return User(id=payload["user_id"])
        return None

    def require_auth(self, handler):
        """Decorator for protected routes."""
        def wrapper(request, *args, **kwargs):
            user = self.authenticate(request)
            if not user:
                return {"error": "Unauthorized"}, 401
            return handler(request, *args, user=user, **kwargs)
        return wrapper
'''

PYTHON_ROUTES = '''"""Route handlers."""

from .auth import AuthMiddleware, create_token

auth = AuthMiddleware()

@auth.require_auth
def get_profile(request, user=None):
    return {"user_id": user.id}

def login(request):
    token = create_token(request.user)
    return {"token": token}
'''

JS_APP = '''import express from 'express';
import { authenticate } from './auth.js';

const app = express();

app.get('/profile', authenticate, (req, res) => {
    res.json({ user: req.user });
});

export default app;
'''


@pytest.fixture
def tools():
    fb = InMemoryFileBackend("/repo", {
        "src/auth.py": PYTHON_AUTH,
        "src/routes.py": PYTHON_ROUTES,
        "src/app.js": JS_APP,
        "config.yaml": "secret: changeme\nport: 8080\n",
    })
    smart = SmartFileAccess(fb, ASTFileAnalysisBackend())
    return {t.name: t for t in create_progressive_tools(smart)}


class TestProgressiveDisclosureFlow:
    """Simulate the agent's progressive exploration pattern."""

    def test_outline_then_symbol_flow(self, tools):
        """Step 1: Outline a file. Step 2: Read specific symbol."""
        # Outline first — cheap
        outline = tools["get_file_outline"].invoke({"file_path": "src/auth.py"})
        assert outline["language"] == "python"
        assert any(s["name"] == "create_token" for s in outline["symbols"])
        assert any(s["name"] == "AuthMiddleware" for s in outline["symbols"])

        # Drill into specific symbol — targeted
        detail = tools["read_symbol"].invoke({"file_path": "src/auth.py", "symbol_name": "create_token"})
        assert "jwt.encode" in detail["body"]
        assert detail["char_count"] < 500  # much less than 8KB full file

    def test_find_references_flow(self, tools):
        """Step 3: Find who uses a symbol across files."""
        refs = tools["find_references"].invoke({"symbol_name": "create_token"})
        paths = [r["path"] for r in refs["references"]]
        assert "src/routes.py" in paths

    def test_read_file_for_config(self, tools):
        """read_file is appropriate for non-code files."""
        result = tools["read_file"].invoke({"file_path": "config.yaml"})
        assert "secret: changeme" in result["content"]

    def test_total_context_size_is_small(self, tools):
        """Verify the progressive approach produces much less data than full reads."""
        # Simulate a realistic exploration session
        total_bytes = 0

        # Outline 3 files
        for path in ["src/auth.py", "src/routes.py", "src/app.js"]:
            result = tools["get_file_outline"].invoke({"file_path": path})
            total_bytes += len(str(result))

        # Read 2 specific symbols
        for sym in [("src/auth.py", "create_token"), ("src/auth.py", "verify_token")]:
            result = tools["read_symbol"].invoke({"file_path": sym[0], "symbol_name": sym[1]})
            total_bytes += len(str(result))

        # Find references for 1 symbol
        result = tools["find_references"].invoke({"symbol_name": "authenticate"})
        total_bytes += len(str(result))

        # Total should be well under 8KB (a single full file read)
        assert total_bytes < 8000, f"Progressive flow produced {total_bytes} bytes — should be under 8KB"
