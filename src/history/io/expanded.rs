use crate::history::{History, Snapshot};
use chrono::{TimeZone, Utc};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::LazyLock;

pub fn read_expanded<A: std::str::FromStr + Ord, P: AsRef<Path>>(
    input: P,
    included_ids: Option<&BTreeSet<A>>,
) -> Result<History<A>, super::Error> {
    static FILE_NAME_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(\d+)-(\d+)\.txt").unwrap());

    let files = std::fs::read_dir(input)?
        .map(|entry| {
            let entry = entry?;
            let path = entry.path();
            let (id, timestamp) = path
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .and_then(|file_name| FILE_NAME_RE.captures(file_name))
                .and_then(|captures| captures.get(1).zip(captures.get(2)))
                .and_then(|(id_match, timestamp_match)| {
                    id_match.as_str().parse::<A>().ok().zip(
                        timestamp_match
                            .as_str()
                            .parse::<i64>()
                            .ok()
                            .and_then(|timestamp| Utc.timestamp_opt(timestamp, 0).single()),
                    )
                })
                .ok_or_else(|| super::Error::InvalidFileName(path.clone()))?;

            Ok((path, id, timestamp))
        })
        .filter_map(|result| {
            result.map_or_else(
                |error| Some(Err(error)),
                |(path, id, timestamp)| {
                    if included_ids.is_none_or(|ids| ids.contains(&id)) {
                        std::fs::File::open(path)
                            .map_err(super::Error::from)
                            .and_then(|file| {
                                BufReader::new(file)
                                    .lines()
                                    .map(|line| {
                                        line.map_err(super::Error::from).and_then(|line| {
                                            line.parse::<A>()
                                                .map_err(|_| super::Error::InvalidLine(line))
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()
                            })
                            .map_or_else(
                                |error| Some(Err(error)),
                                |values| Some(Ok((id, (timestamp, values)))),
                            )
                    } else {
                        None
                    }
                },
            )
        })
        .collect::<Result<Vec<_>, super::Error>>()?;

    let mut result = BTreeMap::new();

    for (id, (timestamp, values)) in files {
        let entry: &mut Vec<_> = result.entry(id).or_default();
        entry.push(Snapshot::new(timestamp, values));
    }

    for values in result.values_mut() {
        values.sort();
    }

    Ok(History::new(result))
}

pub fn write_expanded<A: Copy + std::fmt::Display, P: AsRef<Path>>(
    history: &History<A>,
    output: P,
) -> Result<(), super::Error> {
    for (id, snapshots) in history.values() {
        for Snapshot { timestamp, ids } in snapshots {
            let mut writer = BufWriter::new(File::create(output.as_ref().join(format!(
                "{}-{}.txt",
                id,
                timestamp.timestamp()
            )))?);

            for id in ids {
                writeln!(writer, "{id}")?;
            }
        }
    }

    Ok(())
}
