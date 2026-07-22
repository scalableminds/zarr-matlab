use std::env;

fn main() {
    // Re-run the build script whenever the link paths change.
    println!("cargo:rerun-if-env-changed=EXTRALINKPATHS");

    let link_paths = env::var("EXTRALINKPATHS").unwrap_or_default();

    for link_path in link_paths.split(';') {
        // Skip empty entries (e.g. when EXTRALINKPATHS is unset), which would
        // otherwise cause rustc to error with "empty search path given via `-L`".
        if link_path.is_empty() {
            continue;
        }
        println!("cargo:rustc-link-search={}", link_path);
    }
}
