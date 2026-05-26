use blake3::Hasher;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyConstraintKindV1 {
    MaxRecursionDepth,
    NoGatewayAction,
    NoPolicyMutation,
    NoIdentityFinalization,
    NoEvidenceArchiveAppend,
    NoRuntimeScheduler,
    ReadOnlyPolicyField,
}

impl PolicyConstraintKindV1 {
    fn as_u8(self) -> u8 {
        match self {
            Self::MaxRecursionDepth => 1,
            Self::NoGatewayAction => 2,
            Self::NoPolicyMutation => 3,
            Self::NoIdentityFinalization => 4,
            Self::NoEvidenceArchiveAppend => 5,
            Self::NoRuntimeScheduler => 6,
            Self::ReadOnlyPolicyField => 7,
        }
    }

    fn is_required(self) -> bool {
        matches!(
            self,
            Self::NoGatewayAction
                | Self::NoPolicyMutation
                | Self::NoIdentityFinalization
                | Self::NoEvidenceArchiveAppend
                | Self::NoRuntimeScheduler
                | Self::ReadOnlyPolicyField
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyConstraintV1 {
    pub id: &'static str,
    pub kind: PolicyConstraintKindV1,
    pub bound: Option<u32>,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyFieldV1 {
    pub field_id: &'static str,
    pub version: u32,
    pub constraints: Vec<PolicyConstraintV1>,
    pub read_only: bool,
    pub lower_layer_writable: bool,
    pub gateway_authority: bool,
    pub action_authority: bool,
    pub identity_authority: bool,
    pub evidence_archive_authority: bool,
    pub runtime_enforcement: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFieldErrorV1 {
    EmptyFieldId,
    EmptyConstraintId,
    ZeroVersion,
    NoConstraints,
    MutablePolicyField,
    LowerLayerWritable,
    GatewayAuthority,
    ActionAuthority,
    IdentityAuthority,
    EvidenceArchiveAuthority,
    RuntimeEnforcementAuthority,
    DisabledRequiredConstraint,
}

impl PolicyFieldV1 {
    pub fn validate(&self) -> Result<(), PolicyFieldErrorV1> {
        if self.field_id.is_empty() {
            return Err(PolicyFieldErrorV1::EmptyFieldId);
        }
        if self.version == 0 {
            return Err(PolicyFieldErrorV1::ZeroVersion);
        }
        if self.constraints.is_empty() {
            return Err(PolicyFieldErrorV1::NoConstraints);
        }
        if !self.read_only {
            return Err(PolicyFieldErrorV1::MutablePolicyField);
        }
        if self.lower_layer_writable {
            return Err(PolicyFieldErrorV1::LowerLayerWritable);
        }
        if self.gateway_authority {
            return Err(PolicyFieldErrorV1::GatewayAuthority);
        }
        if self.action_authority {
            return Err(PolicyFieldErrorV1::ActionAuthority);
        }
        if self.identity_authority {
            return Err(PolicyFieldErrorV1::IdentityAuthority);
        }
        if self.evidence_archive_authority {
            return Err(PolicyFieldErrorV1::EvidenceArchiveAuthority);
        }
        if self.runtime_enforcement {
            return Err(PolicyFieldErrorV1::RuntimeEnforcementAuthority);
        }

        for constraint in &self.constraints {
            if constraint.id.is_empty() {
                return Err(PolicyFieldErrorV1::EmptyConstraintId);
            }
            if constraint.kind.is_required() && !constraint.enabled {
                return Err(PolicyFieldErrorV1::DisabledRequiredConstraint);
            }
        }

        Ok(())
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ucf.policy_ecology.policy_field_v1\0");
        write_str(&mut out, self.field_id);
        write_u32(&mut out, self.version);
        write_bool(&mut out, self.read_only);
        write_bool(&mut out, self.lower_layer_writable);
        write_bool(&mut out, self.gateway_authority);
        write_bool(&mut out, self.action_authority);
        write_bool(&mut out, self.identity_authority);
        write_bool(&mut out, self.evidence_archive_authority);
        write_bool(&mut out, self.runtime_enforcement);
        write_u32(&mut out, self.constraints.len() as u32);

        for constraint in &self.constraints {
            write_str(&mut out, constraint.id);
            out.push(constraint.kind.as_u8());
            match constraint.bound {
                Some(v) => {
                    out.push(1);
                    write_u32(&mut out, v);
                }
                None => out.push(0),
            }
            write_bool(&mut out, constraint.enabled);
        }

        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.deterministic_bytes());
        *hasher.finalize().as_bytes()
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    pub fn allows_lower_layer_write(&self) -> bool {
        self.lower_layer_writable
    }

    pub fn has_action_authority(&self) -> bool {
        self.action_authority
    }
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn write_str(out: &mut Vec<u8>, value: &str) {
    write_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}
