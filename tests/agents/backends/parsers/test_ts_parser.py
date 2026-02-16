"""Tests for JS/TS tree-sitter parser."""

import pytest
from src.agents.backends.parsers.ts_parser import TSParser


SAMPLE_JS = '''import { Router } from 'express';
import jwt from 'jsonwebtoken';

const SECRET = 'abc123';

/**
 * Authenticate a user token.
 */
function authenticate(token) {
    const payload = jwt.verify(token, SECRET);
    return payload;
}

class AuthHandler {
    constructor(secret) {
        this.secret = secret;
    }

    validate(token) {
        return jwt.verify(token, this.secret);
    }

    refresh(user) {
        return jwt.sign({ id: user.id }, this.secret);
    }
}

const helper = (x) => x + 1;

export default AuthHandler;
'''

SAMPLE_TS = '''import { Request, Response } from 'express';

interface UserPayload {
    id: string;
    email: string;
}

export function verifyToken(token: string): UserPayload {
    return JSON.parse(atob(token));
}

export class TokenService {
    private secret: string;

    constructor(secret: string) {
        this.secret = secret;
    }

    sign(payload: UserPayload): string {
        return btoa(JSON.stringify(payload));
    }
}
'''


class TestTSParserJS:
    def setup_method(self):
        self.parser = TSParser()

    def test_finds_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        names = [s.name for s in symbols]
        assert "authenticate" in names

    def test_finds_classes(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "AuthHandler"

    def test_class_has_method_children(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        cls = next(s for s in symbols if s.name == "AuthHandler")
        child_names = [c.name for c in cls.children]
        assert "validate" in child_names
        assert "refresh" in child_names

    def test_finds_arrow_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        names = [s.name for s in symbols]
        assert "helper" in names

    def test_finds_constants(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        names = [s.name for s in symbols]
        assert "SECRET" in names

    def test_captures_line_numbers(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        func = next(s for s in symbols if s.name == "authenticate")
        assert func.line > 0
        assert func.line_end >= func.line


class TestTSParserTS:
    def setup_method(self):
        self.parser = TSParser()

    def test_finds_ts_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_TS, "app.ts")
        names = [s.name for s in symbols]
        assert "verifyToken" in names

    def test_finds_ts_classes(self):
        symbols = self.parser.get_symbols(SAMPLE_TS, "app.ts")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "TokenService"

    def test_finds_ts_interfaces(self):
        symbols = self.parser.get_symbols(SAMPLE_TS, "app.ts")
        interfaces = [s for s in symbols if s.kind == "interface"]
        assert len(interfaces) == 1
        assert interfaces[0].name == "UserPayload"


class TestTSParserImports:
    def setup_method(self):
        self.parser = TSParser()

    def test_js_imports(self):
        imports = self.parser.get_imports(SAMPLE_JS)
        assert "express" in imports
        assert "jsonwebtoken" in imports

    def test_ts_imports(self):
        imports = self.parser.get_imports(SAMPLE_TS)
        assert "express" in imports


class TestTSParserExtractSymbol:
    def setup_method(self):
        self.parser = TSParser()

    def test_extracts_function(self):
        detail = self.parser.extract_symbol(SAMPLE_JS, "authenticate", "app.js")
        assert detail is not None
        assert "jwt.verify" in detail.body
        assert detail.char_count > 0

    def test_extracts_method(self):
        detail = self.parser.extract_symbol(SAMPLE_JS, "validate", "app.js")
        assert detail is not None
        assert detail.parent == "AuthHandler"

    def test_returns_none_for_missing(self):
        detail = self.parser.extract_symbol(SAMPLE_JS, "nonexistent", "app.js")
        assert detail is None
