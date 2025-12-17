#![forbid(unsafe_code)]

extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree};

fn extract_ident(input: &TokenStream) -> Option<String> {
    let mut capture_next = false;
    for token in input.clone() {
        match token {
            TokenTree::Ident(ident) => {
                let ident_str = ident.to_string();
                if capture_next {
                    return Some(ident_str);
                }
                if ident_str == "struct" || ident_str == "enum" {
                    capture_next = true;
                }
            }
            _ => {}
        }
    }
    None
}

#[proc_macro_derive(Error, attributes(error, from, source))]
pub fn derive_error(input: TokenStream) -> TokenStream {
    let name = extract_ident(&input).expect("expected an identifier for Error derive");
    let display_impl = format!(
        "impl std::fmt::Display for {name} {{\n    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{\n        write!(f, \"{name}\")\n    }}\n}}\n"
    );
    let error_impl = format!("impl std::error::Error for {name} {{}}\n");
    let expanded = format!("{display_impl}{error_impl}");
    expanded.parse().expect("failed to build Error impl")
}
