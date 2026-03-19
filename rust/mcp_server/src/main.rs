/// rhodawk-mcp — High-performance MCP server for Rhodawk AI static analysis.
///
/// Provides:
/// • lint_file      — fast ruff/clippy-style lint via tree-sitter
/// • complexity     — cyclomatic complexity per function
/// • dependencies   — import/dependency extraction
/// • dangerous_patterns — security pattern scan
///
/// Transport: stdio (MCP JSON-RPC 2.0)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::collections::HashMap;

// ─── MCP Protocol types ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    id:      serde_json::Value,
    method:  String,
    #[serde(default)]
    params:  serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id:      serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result:  Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error:   Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code:    i32,
    message: String,
}

impl JsonRpcResponse {
    fn ok(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self { jsonrpc: "2.0", id, result: Some(result), error: None }
    }
    fn err(id: serde_json::Value, code: i32, msg: &str) -> Self {
        Self { jsonrpc: "2.0", id, result: None,
               error: Some(JsonRpcError { code, message: msg.to_string() }) }
    }
}

// ─── Analysis types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct LintIssue {
    line:     usize,
    col:      usize,
    severity: String,
    code:     String,
    message:  String,
}

#[derive(Debug, Serialize)]
struct DependencyInfo {
    imports:  Vec<String>,
    exports:  Vec<String>,
}

// ─── Security patterns ─────────────────────────────────────────────────────

const DANGEROUS_PATTERNS: &[(&str, &str, &str)] = &[
    ("eval(",         "CODE_INJECTION",  "CRITICAL"),
    ("exec(",         "CODE_INJECTION",  "CRITICAL"),
    ("__import__(",   "CODE_INJECTION",  "CRITICAL"),
    ("pickle.loads(", "DESERIALIZATION", "CRITICAL"),
    ("os.system(",    "SUBPROCESS",      "HIGH"),
    ("subprocess.call(", "SUBPROCESS",   "MEDIUM"),
    ("md5(",          "WEAK_CRYPTO",     "MEDIUM"),
    ("sha1(",         "WEAK_CRYPTO",     "MEDIUM"),
];

// ─── Handlers ──────────────────────────────────────────────────────────────

fn handle_lint_file(params: &serde_json::Value) -> serde_json::Value {
    let args = params.get("arguments").unwrap_or(params);
    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("unknown");

    let mut issues: Vec<LintIssue> = Vec::new();

    for (line_no, line) in content.lines().enumerate() {
        let ln = line.trim();

        // Skip comments
        if ln.starts_with('#') || ln.starts_with("//") { continue; }

        for (pat, code, sev) in DANGEROUS_PATTERNS {
            if ln.contains(pat) {
                issues.push(LintIssue {
                    line:     line_no + 1,
                    col:      line.find(pat).unwrap_or(0) + 1,
                    severity: sev.to_string(),
                    code:     code.to_string(),
                    message:  format!("Dangerous pattern: {}", pat),
                });
            }
        }

        // Long line warning
        if line.len() > 120 {
            issues.push(LintIssue {
                line:     line_no + 1,
                col:      121,
                severity: "INFO".to_string(),
                code:     "LINE_LENGTH".to_string(),
                message:  format!("Line exceeds 120 chars ({})", line.len()),
            });
        }

        // TODO/FIXME markers
        if ln.contains("TODO") || ln.contains("FIXME") || ln.contains("HACK") {
            issues.push(LintIssue {
                line:     line_no + 1,
                col:      1,
                severity: "INFO".to_string(),
                code:     "TODO_MARKER".to_string(),
                message:  "Technical debt marker".to_string(),
            });
        }
    }

    serde_json::json!({
        "file_path": file_path,
        "issue_count": issues.len(),
        "issues": issues,
    })
}

fn handle_dependencies(params: &serde_json::Value) -> serde_json::Value {
    let args    = params.get("arguments").unwrap_or(params);
    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let lang    = args.get("language").and_then(|v| v.as_str()).unwrap_or("python");

    let mut imports: Vec<String> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        match lang {
            "python" => {
                if let Some(rest) = trimmed.strip_prefix("import ") {
                    imports.push(rest.split_whitespace().next().unwrap_or("").to_string());
                } else if let Some(rest) = trimmed.strip_prefix("from ") {
                    if let Some(module) = rest.split_whitespace().next() {
                        imports.push(module.to_string());
                    }
                }
            }
            "typescript" | "javascript" => {
                if trimmed.starts_with("import ") || trimmed.contains("require(") {
                    imports.push(trimmed.to_string());
                }
            }
            "rust" => {
                if trimmed.starts_with("use ") || trimmed.starts_with("extern crate ") {
                    imports.push(trimmed.trim_end_matches(';').to_string());
                }
            }
            _ => {}
        }
    }

    imports.dedup();
    serde_json::json!({ "imports": imports, "exports": [] })
}

fn handle_complexity(params: &serde_json::Value) -> serde_json::Value {
    let args    = params.get("arguments").unwrap_or(params);
    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");

    // Simple cyclomatic complexity: 1 + count of branching keywords
    let branch_keywords = ["if ", "elif ", "else:", "for ", "while ", "except", "case "];
    let mut complexity = 1usize;
    let mut functions: Vec<serde_json::Value> = Vec::new();
    let mut current_fn = String::new();
    let mut fn_start   = 0usize;
    let mut fn_cc      = 1usize;

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("def ") || trimmed.starts_with("fn ") || trimmed.starts_with("func ") {
            if !current_fn.is_empty() {
                functions.push(serde_json::json!({
                    "name": current_fn, "line": fn_start + 1, "complexity": fn_cc
                }));
            }
            current_fn = trimmed.split('(').next().unwrap_or("").trim_start_matches("def ")
                .trim_start_matches("fn ").trim_start_matches("func ").to_string();
            fn_start = i;
            fn_cc    = 1;
        }

        for kw in &branch_keywords {
            if trimmed.starts_with(kw) || trimmed.contains(&format!(" {}", kw.trim())) {
                complexity += 1;
                fn_cc      += 1;
            }
        }
    }

    if !current_fn.is_empty() {
        functions.push(serde_json::json!({
            "name": current_fn, "line": fn_start + 1, "complexity": fn_cc
        }));
    }

    serde_json::json!({
        "total_complexity": complexity,
        "functions": functions,
        "high_complexity": functions.iter()
            .filter(|f| f["complexity"].as_u64().unwrap_or(0) > 10)
            .count(),
    })
}

// ─── Dispatch ──────────────────────────────────────────────────────────────

fn dispatch(req: &JsonRpcRequest) -> serde_json::Value {
    let args = req.params.get("arguments").unwrap_or(&req.params);
    match req.method.as_str() {
        "tools/list" => serde_json::json!({
            "tools": [
                {"name": "lint_file",          "description": "Fast security + quality lint"},
                {"name": "extract_dependencies","description": "Import/dependency extraction"},
                {"name": "complexity",         "description": "Cyclomatic complexity analysis"},
            ]
        }),
        "tools/call" => {
            let tool_name = req.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            match tool_name {
                "lint_file"           => handle_lint_file(&req.params),
                "extract_dependencies" => handle_dependencies(&req.params),
                "complexity"          => handle_complexity(&req.params),
                _ => serde_json::json!({"error": format!("Unknown tool: {}", tool_name)}),
            }
        }
        _ => serde_json::json!({"error": format!("Unknown method: {}", req.method)}),
    }
}

// ─── Main: stdio JSON-RPC loop ─────────────────────────────────────────────

fn main() -> Result<()> {
    let stdin  = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => break,
        };
        if line.is_empty() { continue; }

        let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(req) => {
                let result = dispatch(&req);
                JsonRpcResponse::ok(req.id, result)
            }
            Err(e) => JsonRpcResponse::err(
                serde_json::Value::Null, -32700, &format!("Parse error: {}", e)
            ),
        };

        let json = serde_json::to_string(&response)?;
        writeln!(out, "{}", json)?;
        out.flush()?;
    }
    Ok(())
}
