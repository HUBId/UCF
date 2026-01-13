#![forbid(unsafe_code)]

pub trait DigestAlgo {
    fn algorithm(&self) -> &str;
    fn digest(&self, input: &[u8]) -> Vec<u8>;
}

pub trait VrfLike {
    fn algorithm(&self) -> &str;
    fn prove(&self, input: &[u8]) -> (Vec<u8>, Vec<u8>);
}

pub trait SignatureLike {
    fn algorithm(&self) -> &str;
    fn sign(&self, input: &[u8]) -> Vec<u8>;
    fn verify(&self, input: &[u8], signature: &[u8]) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopDigest;

    impl DigestAlgo for NoopDigest {
        fn algorithm(&self) -> &str {
            "noop"
        }

        fn digest(&self, input: &[u8]) -> Vec<u8> {
            input.to_vec()
        }
    }

    struct NoopVrf;

    impl VrfLike for NoopVrf {
        fn algorithm(&self) -> &str {
            "noop-vrf"
        }

        fn prove(&self, input: &[u8]) -> (Vec<u8>, Vec<u8>) {
            (input.to_vec(), vec![0u8; input.len()])
        }
    }

    struct NoopSignature;

    impl SignatureLike for NoopSignature {
        fn algorithm(&self) -> &str {
            "noop-signature"
        }

        fn sign(&self, input: &[u8]) -> Vec<u8> {
            input.to_vec()
        }

        fn verify(&self, input: &[u8], signature: &[u8]) -> bool {
            input == signature
        }
    }

    #[test]
    fn noop_digest_roundtrip() {
        let digest = NoopDigest;
        assert_eq!(digest.algorithm(), "noop");
        assert_eq!(digest.digest(b"data"), b"data".to_vec());
    }

    #[test]
    fn noop_vrf_outputs() {
        let vrf = NoopVrf;
        let (proof, output) = vrf.prove(b"seed");
        assert_eq!(proof, b"seed".to_vec());
        assert_eq!(output, vec![0u8; 4]);
    }

    #[test]
    fn noop_signature_verifies() {
        let signature = NoopSignature;
        let sig = signature.sign(b"payload");
        assert!(signature.verify(b"payload", &sig));
        assert!(!signature.verify(b"payload", b"bad"));
    }
}
