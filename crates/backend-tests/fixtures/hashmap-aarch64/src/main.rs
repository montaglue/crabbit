//! HashMap end-to-end: `HashMap::new` seeds its `RandomState` from std's
//! `#[thread_local]` `KEYS` static, so constructing, filling, and reading a
//! map exercises the backend's ELF TLS path (local-exec `tpidr_el0`
//! addressing against libstd's `.tdata`) on top of hashing, probing, and
//! resize.

use std::collections::HashMap;

pub fn make_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("crab".to_string(), "bit".to_string());
    map.insert("thread".to_string(), "local".to_string());
    map
}

fn main() {
    let mut map = make_map();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get("crab").map(String::as_str), Some("bit"));
    assert_eq!(map.get("thread").map(String::as_str), Some("local"));
    assert_eq!(map.get("missing"), None);

    // Overwrite returns the old value through the same probe path.
    let old = map.insert("crab".to_string(), "bitte".to_string());
    assert_eq!(old.as_deref(), Some("bit"));
    assert_eq!(map.get("crab").map(String::as_str), Some("bitte"));

    // Enough inserts to force at least one resize/rehash.
    let mut numbers: HashMap<u64, u64> = HashMap::new();
    for i in 0..100u64 {
        numbers.insert(i, i * i);
    }
    assert_eq!(numbers.len(), 100);
    let mut sum = 0u64;
    for i in 0..100u64 {
        sum += numbers[&i];
    }
    assert_eq!(sum, (0..100u64).map(|i| i * i).sum::<u64>());
    assert!(numbers.remove(&42).is_some());
    assert_eq!(numbers.get(&42), None);
    assert_eq!(numbers.len(), 99);

    println!("hashmap ok len={} crab={}", map.len(), map["crab"]);
}
