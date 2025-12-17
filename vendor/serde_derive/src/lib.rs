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

#[proc_macro_derive(Serialize)]
pub fn derive_serialize(input: TokenStream) -> TokenStream {
    let name = extract_ident(&input).expect("expected an identifier for Serialize derive");
    let expanded = format!("impl serde::Serialize for {name} {{}}",);
    expanded.parse().expect("failed to build Serialize impl")
}

#[proc_macro_derive(Deserialize)]
pub fn derive_deserialize(input: TokenStream) -> TokenStream {
    let name = extract_ident(&input).expect("expected an identifier for Deserialize derive");
    let expanded = format!("impl<'de> serde::Deserialize<'de> for {name} {{}}",);
    expanded.parse().expect("failed to build Deserialize impl")
}
