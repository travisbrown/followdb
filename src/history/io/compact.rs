use crate::diff::io::{ReadExt, WriteExt};
use crate::history::{History, Snapshot};
use base64::{Engine, engine::general_purpose::STANDARD};
use std::collections::{BTreeMap, btree_map::Entry};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Cursor, Write};
use std::path::Path;

/// Read a history from a compact binary file.
///
/// # Panics
///
/// Panics if an internal map invariant is violated (an occupied entry has no snapshots), which
/// should never occur in practice.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read, if any update cannot be decoded, or
/// if a diff cannot be applied to reconstruct a snapshot.
pub fn read_compact<P: AsRef<Path>>(input: P) -> Result<History<u64>, super::Error> {
    let mut result = BTreeMap::new();
    let mut buffer = vec![];

    for line in BufReader::new(File::open(input)?).lines() {
        let line = line?;
        buffer.clear();
        STANDARD.decode_vec(line, &mut buffer)?;
        let len = buffer.len();

        let mut cursor = Cursor::new(buffer);

        while cursor.position() < len as u64 {
            let update = cursor.read_update()?;

            match result.entry(update.id) {
                Entry::Vacant(entry) => {
                    entry.insert(vec![Snapshot::new(
                        update.timestamp,
                        update.diff.update(&[])?,
                    )]);
                }
                Entry::Occupied(mut entry) => {
                    let entry = entry.get_mut();
                    // Safe because we always initialize with one value.
                    let previous = entry.last().unwrap();
                    entry.push(Snapshot::new(
                        update.timestamp,
                        update.diff.update(previous.ids.as_slice())?,
                    ));
                }
            }
        }

        buffer = cursor.into_inner();
    }

    Ok(History::new(result))
}

/// Write a history to a compact binary file.
///
/// # Errors
///
/// Returns an error if any update cannot be encoded or if the output file cannot be written.
pub fn write_compact<P: AsRef<Path>>(
    history: &History<u64>,
    output: P,
) -> Result<(), super::Error> {
    let mut lines = vec![];
    let mut buffer = vec![];

    for update in history.updates() {
        let (_, update) = update?;
        buffer.clear();
        buffer.write_update(&update)?;
        let line = STANDARD.encode(&buffer);
        lines.push((update.timestamp, update.id, line));
    }

    lines.sort();

    let mut writer = BufWriter::new(File::create(output)?);

    for (_, _, line) in lines {
        writeln!(writer, "{line}")?;
    }

    Ok(())
}
