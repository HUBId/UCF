# ucf-iit-monitor

This crate provides a deterministic, **proxy** coupling metric between module digests. It is **not** a real IIT/Phi implementation. The score derives from a stable hash mapping into `0..=10000` to give a repeatable integration/coupling proxy for experiments. A real IIT/Phi implementation would replace this proxy with a full cause-effect repertoire computation and proper Phi maximization.
