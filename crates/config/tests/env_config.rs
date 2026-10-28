//! `with_env_config`: the resident server's way to run one configuration.
//! Its own test binary: it mutates the process environment, which no other
//! test may read concurrently.

use std::collections::BTreeMap;

use crabbit_config::{ConfigError, with_env_config};

#[test]
fn rejects_non_crabbit_keys_and_restores_the_environment() {
    let mut bad = BTreeMap::new();
    bad.insert("PATH".to_string(), "/nowhere".to_string());
    assert!(matches!(
        with_env_config(&bad, || ()),
        Err(ConfigError::NonCrabbitKey { key }) if key == "PATH"
    ));

    let mut config = BTreeMap::new();
    config.insert("CRABBIT_ENV_CONFIG_TEST".to_string(), "on".to_string());
    with_env_config(&config, || {
        assert_eq!(
            std::env::var("CRABBIT_ENV_CONFIG_TEST").as_deref(),
            Ok("on")
        );
    })
    .unwrap();
    assert!(
        std::env::var("CRABBIT_ENV_CONFIG_TEST").is_err(),
        "restored after the run"
    );
}
