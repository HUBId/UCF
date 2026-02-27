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
fn size_caps_and_auth_denials_are_enforced_and_logged() {
    let td = tempdir().expect("tmp");
    let mut svc = GatewayService::new(GatewayConfig::for_tests(td.path()));

    let mut big = submit_request([1, 2, 3, 4, 5, 6, 7, 8]);
    big.payload_text_utf8 = vec![b'x'; 5000];
    let err = svc
        .submit_control_frame("test-token", big)
        .expect_err("must reject");
    assert!(err.to_string().contains("large"));

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
    assert_eq!(err.to_string(), "unauthorized");

    let _ = svc.handle_request(
        proto::GatewayRequest {
            negotiated_version: 1,
            payload: Some(proto::gateway_request::Payload::Handshake(
                proto::HandshakeRequest {
                    schema_version: 1,
                    client_id: "client-a".to_string(),
                    supported_versions: vec![1],
                    auth_token: "bad".to_string(),
                },
            )),
        },
        "bad",
        "client-a",
    );

    let log_body =
        std::fs::read_to_string(td.path().join("gateway_access_records.jsonl")).expect("log");
    assert!(log_body.contains("\"endpoint\":\"handshake\""));
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
        let req = read_frame(&mut socket, 128 * 1024).expect("read");
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
