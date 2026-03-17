//! A history is a collection of timestamped snapshots of follow lists for a set of accounts.
use crate::diff::{Diff, Update};
use chrono::{DateTime, Utc};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

pub mod io;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Action {
    #[serde(rename = "+")]
    Add,
    #[serde(rename = "-")]
    Remove,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Change<A> {
    pub source_id: A,
    pub target_id: A,
    pub action: Action,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub after: DateTime<Utc>,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub before: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Snapshot<A> {
    pub timestamp: DateTime<Utc>,
    pub ids: Vec<A>,
}

impl<A> Snapshot<A> {
    #[must_use]
    pub const fn new(timestamp: DateTime<Utc>, ids: Vec<A>) -> Self {
        Self { timestamp, ids }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct History<A>(BTreeMap<A, Vec<Snapshot<A>>>);

impl<A> History<A> {
    #[must_use]
    pub const fn new(values: BTreeMap<A, Vec<Snapshot<A>>>) -> Self {
        Self(values)
    }
}

impl<A: Copy> History<A> {
    pub fn account_ids(&self) -> impl Iterator<Item = A> {
        self.0.iter().flat_map(|(id, snapshots)| {
            std::iter::once(*id).chain(
                snapshots
                    .iter()
                    .flat_map(|snapshot| snapshot.ids.iter().copied()),
            )
        })
    }

    pub fn values(&self) -> impl Iterator<Item = (A, &[Snapshot<A>])> {
        self.0
            .iter()
            .map(|(id, snapshots)| (*id, snapshots.as_slice()))
    }
}

impl<A: Copy + Eq + Ord> History<A> {
    pub fn updates(
        &self,
    ) -> impl Iterator<Item = Result<(Option<DateTime<Utc>>, Update<A>), crate::diff::error::InputError>>
    {
        self.0
            .iter()
            .filter_map(|(id, snapshots)| {
                snapshots.first().map(|Snapshot { timestamp, ids }| {
                    std::iter::once(Ok((
                        None,
                        Update::new(*timestamp, *id, Diff::init(ids.clone())),
                    )))
                    // Safe because we confirm above that `snapshots` is non-empty.
                    .chain(snapshots.windows(2).map(|window| match window {
                        [
                            Snapshot {
                                timestamp: a_timestamp,
                                ids: a_ids,
                            },
                            Snapshot {
                                timestamp: b_timestamp,
                                ids: b_ids,
                            },
                        ] => Ok((
                            Some(*a_timestamp),
                            Update::new(*b_timestamp, *id, Diff::compute(a_ids, b_ids)?),
                        )),
                        // Safe because of the contract of `windows`.
                        _ => panic!("Invalid snapshot window"),
                    }))
                })
            })
            .flatten()
    }

    #[must_use]
    pub fn changes(&self) -> Vec<Change<A>> {
        let mut changes: Vec<Change<A>> = self
            .0
            .iter()
            .filter(|(_, snapshots)| snapshots.len() > 1)
            .flat_map(|(id, snapshots)| {
                let mut changes = vec![];
                let mut previous_timestamp = snapshots[0].timestamp;
                let mut previous_ids = snapshots[0].ids.iter().copied().collect::<BTreeSet<_>>();

                for Snapshot { timestamp, ids } in &snapshots[1..] {
                    let ids = ids.iter().copied().collect::<BTreeSet<_>>();

                    for removed_id in previous_ids.difference(&ids) {
                        changes.push(Change {
                            source_id: *id,
                            target_id: *removed_id,
                            action: Action::Remove,
                            after: previous_timestamp,
                            before: *timestamp,
                        });
                    }

                    for added_id in ids.difference(&previous_ids) {
                        changes.push(Change {
                            source_id: *id,
                            target_id: *added_id,
                            action: Action::Add,
                            after: previous_timestamp,
                            before: *timestamp,
                        });
                    }

                    previous_timestamp = *timestamp;
                    previous_ids = ids;
                }

                changes
            })
            .collect();

        changes.sort_by_key(|change| (Reverse(change.before), change.source_id));

        changes
    }
}
