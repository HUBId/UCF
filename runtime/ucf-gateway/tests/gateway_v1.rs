use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
use std::thread;

use prost::Message;
use tempfile::tempdir;
use ucf_gateway::proto;
use ucf_gateway::{read_frame, write_frame, GatewayConfig, GatewayService};

fn submit_request(policy: [u8; 8]) -> proto::ControlFrameSubmitRequest {
    proto::ControlFrameSubmitRequest {
        schema_version: 1,
        run_id: "run-test".to_string(),
        policy_graph_digest_prefix: policy.to_vec(),
        tick: 1,
        window: 0,
        corr_id: 42,
        intent_id: 9,
        intent_kind: 1,
        channel: 1,
        intent_summary: "hello".to_string(),
        payload_text_utf8: b"safe text".to_vec(),
        auth_token: "test-token".to_string(),
    }
}

#[test]
fn negotiation_and_submit_and_subscribe_and_ess_query_work() {
    let td = tempdir().expect("tmp");
    let mut svc = GatewayService::new(GatewayConfig::for_tests(td.path()));

    let hs = proto::HandshakeRequest {
        schema_version: 1,
        client_id: "c1".to_string(),
        supported_versions: vec![0, 1],
        auth_token: "test-token".to_string(),
    };
    let hs_resp = svc.negotiate(&hs).expect("negotiate");
    assert_eq!(hs_resp.selected_version, 1);

    let submit = svc
        .submit_control_frame("test-token", submit_request([1, 2, 3, 4, 5, 6, 7, 8]))
        .expect("submit");
    assert!(submit.decision.is_some());

    let sub = svc
        .subscribe_decisions(
            "test-token",
            proto::DecisionStreamSubscribeRequest {
                schema_version: 1,
                run_id: "run-test".to_string(),
                policy_graph_digest_prefix: vec![1, 2, 3, 4, 5, 6, 7, 8],
                max_events: 8,
                auth_token: "test-token".to_string(),
            },
        )
        .expect("subscribe");
    assert_eq!(sub.events.len(), 1);

    let ess = svc
        .query_ess(
            "test-token",
            proto::EssQueryRequest {
                schema_version: 1,
                run_id: "run-test".to_string(),
                policy_graph_digest_prefix: vec![1, 2, 3, 4, 5, 6, 7, 8],
                max_records: 10,
                auth_token: "test-token".to_string(),
            },
        )
        .expect("ess");
    assert!(!ess.summaries.is_empty());
}

#[test]
fn size_caps_auth_denials_rate_limits_and_abuse_logs_are_enforced() {
    let td = tempdir().expect("tmp");
    let mut svc = GatewayService::new(GatewayConfig::for_tests(td.path()));

    let mut big = submit_request([1, 2, 3, 4, 5, 6, 7, 8]);
    big.payload_text_utf8 = vec![b'x'; 5000];
    let err = svc
        .submit_control_frame("test-token", big)
        .expect_err("must reject");
    assert_eq!(err.to_string(), "message too large");

    let err = svc
        .query_ess(
            "wrong-token",
            proto::EssQueryRequest {
                schema_version: 1,
                run_id: "run-test".to_string(),
                policy_graph_digest_prefix: vec![1, 2, 3, 4, 5, 6, 7, 8],
                max_records: 2,
                auth_token: "wrong-token".to_string(),
            },
        )
        .expect_err("must reject auth");
    assert_eq!(err.to_string(), "auth denied");

    for _ in 0..6 {
        let _ = svc.handle_request(
            proto::GatewayRequest {
                negotiated_version: 1,
                payload: Some(proto::gateway_request::Payload::Submit(submit_request([
                    1, 2, 3, 4, 5, 6, 7, 8,
                ]))),
            },
            "test-token",
            "client-a",
        );
    }

    let abuse_body =
        std::fs::read_to_string(td.path().join("gateway_abuse_records.jsonl")).expect("abuse");
    assert!(abuse_body.contains("RateLimit"));
    assert!(!abuse_body.contains("safe text"));

    let deny = svc.handle_request(
        proto::GatewayRequest {
            negotiated_version: 1,
            payload: Some(proto::gateway_request::Payload::EssQuery(
                proto::EssQueryRequest {
                    schema_version: 1,
                    run_id: "run-test".to_string(),
                    policy_graph_digest_prefix: vec![1, 2, 3, 4, 5, 6, 7, 8],
                    max_records: 2,
                    auth_token: "wrong-token".to_string(),
                },
            )),
        },
        "wrong-token",
        "client-a",
    );
    match deny.payload {
        Some(proto::gateway_response::Payload::Error(e)) => {
            assert_eq!(e.code, 1001);
            assert!(!e.request_id.is_empty());
            assert_eq!(e.message, "auth denied");
        }
        _ => panic!("expected error"),
    }
}

#[test]
fn tcp_length_delimited_roundtrip_works() {
    let td = tempdir().expect("tmp");
    let port = 45091;
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);

    let handle = thread::spawn(move || {
        let mut svc = GatewayService::new(GatewayConfig::for_tests(td.path()));
        let listener = std::net::TcpListener::bind(addr).expect("bind");
        let (mut socket, _) = listener.accept().expect("accept");
        let req = read_frame(&mut socket, 256 * 1024).expect("read");
        let resp = svc.handle_request(req, "test-token", "c1");
        write_frame(&mut socket, &resp).expect("write");
    });

    let mut client = (0..20)
        .find_map(|_| {
            TcpStream::connect(addr).ok().or_else(|| {
                std::thread::sleep(std::time::Duration::from_millis(20));
                None
            })
        })
        .expect("connect");
    let req = proto::GatewayRequest {
        negotiated_version: 1,
        payload: Some(proto::gateway_request::Payload::Handshake(
            proto::HandshakeRequest {
                schema_version: 1,
                client_id: "c1".to_string(),
                supported_versions: vec![1],
                auth_token: "test-token".to_string(),
            },
        )),
    };
    let body = req.encode_to_vec();
    client
        .write_all(&(u32::try_from(body.len()).expect("len")).to_le_bytes())
        .expect("len");
    client.write_all(&body).expect("body");

    let mut len = [0u8; 4];
    client.read_exact(&mut len).expect("read len");
    let n = u32::from_le_bytes(len) as usize;
    let mut body = vec![0u8; n];
    client.read_exact(&mut body).expect("read body");
    let resp = proto::GatewayResponse::decode(body.as_slice()).expect("decode");
    match resp.payload {
        Some(proto::gateway_response::Payload::Handshake(h)) => assert_eq!(h.selected_version, 1),
        _ => panic!("wrong payload"),
    }

    handle.join().expect("join");
}

#[test]
fn strict_mode_rejects_non_loopback_tcp_bind_and_records_violation() {
    let td = tempdir().expect("tmp");
    std::env::set_var("UCF_GATEWAY_TRANSPORT", "tcp");
    std::env::set_var("UCF_GATEWAY_BIND", "0.0.0.0:44991");
    std::env::set_var("UCF_STRICT_MODE", "1");
    let err = ucf_gateway::transport_from_env(td.path()).expect_err("reject bind");
    assert_eq!(err.to_string(), "auth denied");
    let body =
        std::fs::read_to_string(td.path().join("network_violations.jsonl")).expect("violation log");
    assert!(body.contains("gateway_non_loopback_bind"));
    std::env::remove_var("UCF_GATEWAY_TRANSPORT");
    std::env::remove_var("UCF_GATEWAY_BIND");
    std::env::remove_var("UCF_STRICT_MODE");
}

#[test]
fn health_endpoint_requires_token_outside_dev_and_returns_bounded_surface() {
    let td = tempdir().expect("tmp");
    let mut svc = GatewayService::new(GatewayConfig::for_tests(td.path()));

    let denied = svc.handle_request(
        proto::GatewayRequest {
            negotiated_version: 1,
            payload: Some(proto::gateway_request::Payload::Health(
                proto::HealthRequest {
                    schema_version: 1,
                    auth_token: "".to_string(),
                },
            )),
        },
        "",
        "hc",
    );
    match denied.payload {
        Some(proto::gateway_response::Payload::Error(e)) => assert_eq!(e.code, 1001),
        _ => panic!("expected auth error"),
    }

    let ok = svc.handle_request(
        proto::GatewayRequest {
            negotiated_version: 1,
            payload: Some(proto::gateway_request::Payload::Health(
                proto::HealthRequest {
                    schema_version: 1,
                    auth_token: "test-token".to_string(),
                },
            )),
        },
        "test-token",
        "hc",
    );
    match ok.payload {
        Some(proto::gateway_response::Payload::Health(h)) => {
            assert_eq!(h.schema_version, 1);
            assert_eq!(h.run_id, "run-test");
            assert!(!h.policy_graph_digest_prefix.is_empty());
            assert!(h.active_slots_summary.chars().count() <= 128);
        }
        _ => panic!("expected health payload"),
    }
}

#[test]
fn gateway_panic_is_caught_and_returns_internal_error() {
    let td = tempdir().expect("tmp");
    let mut svc = GatewayService::new(GatewayConfig::for_tests(td.path()));
    let resp = svc.handle_request(
        proto::GatewayRequest {
            negotiated_version: 1,
            payload: Some(proto::gateway_request::Payload::Handshake(
                proto::HandshakeRequest {
                    schema_version: 1,
                    client_id: "c1".to_string(),
                    supported_versions: vec![1],
                    auth_token: "test-token".to_string(),
                },
            )),
        },
        "test-token",
        "__panic_test__",
    );
    match resp.payload {
        Some(proto::gateway_response::Payload::Error(e)) => {
            assert_eq!(e.code, 1500);
            assert_eq!(e.message, "internal error");
            assert!(!e.request_id.is_empty());
        }
        _ => panic!("expected internal error"),
    }
}
