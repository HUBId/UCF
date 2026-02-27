use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::OpsError;

#[derive(Debug, Clone)]
pub struct ChangeImpactArgs {
    pub base: String,
    pub head: String,
    pub rules_path: PathBuf,
    pub out_md: PathBuf,
    pub out_json: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChangeImpactRules {
    pub max_files: usize,
    pub max_commands: usize,
    pub default_modules: Vec<String>,
    pub default_gates: Vec<String>,
    pub command_catalog: Vec<GateCommand>,
    pub rules: Vec<ChangeRule>,
}

impl Default for ChangeImpactRules {
    fn default() -> Self {
        Self {
            max_files: 500,
            max_commands: 20,
            default_modules: vec!["runtime-core".to_string()],
            default_gates: vec![
                "cargo-test-workspace".to_string(),
                "readiness-gate".to_string(),
            ],
            command_catalog: Vec::new(),
            rules: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCommand {
    pub gate: String,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChangeRule {
    pub name: String,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub modules: Vec<String>,
    pub gates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeImpactPlan {
    pub base: String,
    pub head: String,
    pub total_changed_files: usize,
    pub analyzed_files: usize,
    pub truncated_files: bool,
    pub changed_files: Vec<String>,
    pub affected_modules: Vec<String>,
    pub required_gates: Vec<String>,
    pub commands: Vec<String>,
}

pub fn change_impact(args: &ChangeImpactArgs) -> Result<ChangeImpactPlan, OpsError> {
    let rules = load_rules(&args.rules_path)?;
    let changed = git_changed_files(&args.base, &args.head)?;
    let plan = build_plan(&args.base, &args.head, &changed, &rules);
    write_plan_markdown(&args.out_md, &plan)?;
    write_plan_json(&args.out_json, &plan)?;
    Ok(plan)
}

fn load_rules(path: &Path) -> Result<ChangeImpactRules, OpsError> {
    let raw = fs::read_to_string(path)?;
    let rules = toml::from_str::<ChangeImpactRules>(&raw)
        .map_err(|e| OpsError::Invalid(format!("invalid ruleset {}: {e}", path.display())))?;
    Ok(rules)
}

fn git_changed_files(base: &str, head: &str) -> Result<Vec<String>, OpsError> {
    let output = Command::new("git")
        .arg("diff")
        .arg("--name-only")
        .arg(format!("{base}..{head}"))
        .output()?;
    if !output.status.success() {
        return Err(OpsError::Invalid(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let mut files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

fn build_plan(
    base: &str,
    head: &str,
    changed_files: &[String],
    rules: &ChangeImpactRules,
) -> ChangeImpactPlan {
    let truncated_files = changed_files.len() > rules.max_files;
    let analyzed: Vec<String> = changed_files
        .iter()
        .take(rules.max_files)
        .cloned()
        .collect();

    let mut modules = BTreeSet::new();
    let mut gates = BTreeSet::new();
    let mut known_paths = BTreeSet::new();

    for file in &analyzed {
        let mut matched = false;
        for rule in &rules.rules {
            if rule_matches(rule, file) {
                matched = true;
                known_paths.insert(file.clone());
                for module in &rule.modules {
                    modules.insert(module.clone());
                }
                for gate in &rule.gates {
                    gates.insert(gate.clone());
                }
            }
        }
        if !matched {
            for module in &rules.default_modules {
                modules.insert(module.clone());
            }
            for gate in &rules.default_gates {
                gates.insert(gate.clone());
            }
        }
    }

    if truncated_files {
        for module in &rules.default_modules {
            modules.insert(module.clone());
        }
        for gate in rules.command_catalog.iter().map(|c| c.gate.clone()) {
            gates.insert(gate);
        }
    }

    let catalog = ordered_catalog(&rules.command_catalog);
    let mut commands = Vec::new();
    for gate in &catalog {
        if gates.contains(gate) && commands.len() < rules.max_commands {
            if let Some(command) = rules
                .command_catalog
                .iter()
                .find(|entry| &entry.gate == gate)
                .map(|entry| entry.command.clone())
            {
                commands.push(command);
            }
        }
    }

    ChangeImpactPlan {
        base: base.to_string(),
        head: head.to_string(),
        total_changed_files: changed_files.len(),
        analyzed_files: analyzed.len(),
        truncated_files,
        changed_files: analyzed,
        affected_modules: modules.into_iter().collect(),
        required_gates: gates.into_iter().collect(),
        commands,
    }
}

fn ordered_catalog(catalog: &[GateCommand]) -> Vec<String> {
    let mut order = Vec::new();
    let mut seen = BTreeSet::new();
    for item in catalog {
        if seen.insert(item.gate.clone()) {
            order.push(item.gate.clone());
        }
    }
    order
}

fn rule_matches(rule: &ChangeRule, path: &str) -> bool {
    if !rule.include.is_empty() && !rule.include.iter().any(|pat| glob_matches(pat, path)) {
        return false;
    }
    !rule.exclude.iter().any(|pat| glob_matches(pat, path))
}

fn glob_matches(pattern: &str, text: &str) -> bool {
    let p: Vec<&str> = pattern.split('/').collect();
    let t: Vec<&str> = text.split('/').collect();
    match_segments(&p, &t)
}

fn match_segments(pattern: &[&str], text: &[&str]) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }
    if pattern[0] == "**" {
        if match_segments(&pattern[1..], text) {
            return true;
        }
        if !text.is_empty() {
            return match_segments(pattern, &text[1..]);
        }
        return false;
    }
    if text.is_empty() {
        return false;
    }
    if segment_matches(pattern[0], text[0]) {
        return match_segments(&pattern[1..], &text[1..]);
    }
    false
}

fn segment_matches(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    wildcard_match(pattern.as_bytes(), text.as_bytes())
}

fn wildcard_match(pattern: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t, mut star, mut match_pos) = (0usize, 0usize, None, 0usize);
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == text[t] || pattern[p] == b'?') {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            match_pos = t;
            p += 1;
        } else if let Some(star_idx) = star {
            p = star_idx + 1;
            match_pos += 1;
            t = match_pos;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

fn write_plan_markdown(path: &Path, plan: &ChangeImpactPlan) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str("# Change Impact Plan\n\n");
    out.push_str(&format!("- Base: `{}`\n", plan.base));
    out.push_str(&format!("- Head: `{}`\n", plan.head));
    out.push_str(&format!("- Changed files: {}\n", plan.total_changed_files));
    out.push_str(&format!("- Analyzed files: {}\n", plan.analyzed_files));
    out.push_str(&format!("- Truncated: {}\n\n", plan.truncated_files));

    out.push_str("## Affected modules\n");
    for module in &plan.affected_modules {
        out.push_str(&format!("- {module}\n"));
    }

    out.push_str("\n## Required gates\n");
    for gate in &plan.required_gates {
        out.push_str(&format!("- {gate}\n"));
    }

    out.push_str("\n## Commands\n");
    for command in &plan.commands {
        out.push_str(&format!("- `{command}`\n"));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_plan_json(path: &Path, plan: &ChangeImpactPlan) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(plan)?;
    fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_supports_double_star_and_wildcards() {
        assert!(glob_matches(
            "policies/**",
            "policies/packs/base_v1/file.toml"
        ));
        assert!(glob_matches(
            "runtime/ucf-ops/src/*.rs",
            "runtime/ucf-ops/src/main.rs"
        ));
        assert!(!glob_matches("docs/*.md", "docs/spec/snapshot.md"));
    }

    #[test]
    fn deterministic_plan_and_conservative_defaults() {
        let rules = ChangeImpactRules {
            max_files: 500,
            max_commands: 10,
            default_modules: vec!["runtime-core".to_string()],
            default_gates: vec!["readiness-gate".to_string()],
            command_catalog: vec![
                GateCommand {
                    gate: "cargo-test-workspace".to_string(),
                    command: "cargo test --workspace".to_string(),
                },
                GateCommand {
                    gate: "readiness-gate".to_string(),
                    command: "cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/change_impact/readiness_gate.json".to_string(),
                },
            ],
            rules: vec![ChangeRule {
                name: "policy".to_string(),
                include: vec!["policies/**".to_string()],
                exclude: Vec::new(),
                modules: vec!["ucf-policy".to_string()],
                gates: vec!["cargo-test-workspace".to_string()],
            }],
        };

        let changed = vec!["policies/a.toml".to_string(), "unknown/new.txt".to_string()];
        let plan = build_plan("HEAD~1", "HEAD", &changed, &rules);
        assert_eq!(plan.affected_modules, vec!["runtime-core", "ucf-policy"]);
        assert_eq!(
            plan.required_gates,
            vec!["cargo-test-workspace", "readiness-gate"]
        );
        assert_eq!(
            plan.commands,
            vec![
                "cargo test --workspace",
                "cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/change_impact/readiness_gate.json"
            ]
        );
    }

    #[test]
    fn bounded_command_output() {
        let rules = ChangeImpactRules {
            max_files: 10,
            max_commands: 1,
            default_modules: vec!["runtime-core".to_string()],
            default_gates: vec!["g1".to_string()],
            command_catalog: vec![
                GateCommand {
                    gate: "g1".to_string(),
                    command: "cmd1".to_string(),
                },
                GateCommand {
                    gate: "g2".to_string(),
                    command: "cmd2".to_string(),
                },
            ],
            rules: vec![ChangeRule {
                name: "all".to_string(),
                include: vec!["**".to_string()],
                exclude: Vec::new(),
                modules: vec!["m".to_string()],
                gates: vec!["g1".to_string(), "g2".to_string()],
            }],
        };
        let plan = build_plan("a", "b", &["x.rs".to_string()], &rules);
        assert_eq!(plan.commands, vec!["cmd1"]);
    }
}
