use chrono::{DateTime, Utc};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::ops::Add;

pub mod error;
pub mod io;

/// A single transformation operation.
///
/// Note that the values of `Take` and `Drop` should never be zero, and the sequence of inserted
/// values should never be empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op<A> {
    Take(u64),
    Drop(u64),
    Insert(Vec<A>),
}

impl<A> Op<A> {
    /// Verify that the value is "non-empty".
    ///
    /// If false, then either a take or drop value is zero, or an insert sequence is empty.
    #[must_use]
    pub const fn validate(&self) -> bool {
        match self {
            Self::Take(value) | Self::Drop(value) => *value != 0,
            Self::Insert(values) => !values.is_empty(),
        }
    }

    #[must_use]
    pub const fn same_type(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::Take(_), Self::Take(_))
                | (Self::Drop(_), Self::Drop(_))
                | (Self::Insert(_), Self::Insert(_))
        )
    }
}

/// The operations necessary to transform one list of distinct elements into another.
///
/// The diff implementation is optimized for the case where the order of elements is mostly
/// unchanged (Twitter follower lists, for example), but will produce correct results in the
/// general case.
///
/// Note that there will be no operations if and only if both the source and target lists are
/// empty.
///
/// There will never be two consecutive operations of the same type, and `len_change` should never
/// be larger than `expected_source_len`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diff<A> {
    pub ops: Vec<Op<A>>,
    pub expected_source_len: u64,
    pub len_change: i64,
    /// Stored for convenience only.
    drop_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Update<A> {
    pub timestamp: DateTime<Utc>,
    pub id: A,
    pub diff: Diff<A>,
}

impl<A: Ord> PartialOrd for Update<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self
            .timestamp
            .cmp(&other.timestamp)
            .then(self.id.cmp(&other.id))
        {
            Ordering::Equal => None,
            other => Some(other),
        }
    }
}

impl<A> Update<A> {
    pub const fn new(timestamp: DateTime<Utc>, id: A, diff: Diff<A>) -> Self {
        Self {
            timestamp,
            id,
            diff,
        }
    }
}

impl Update<u64> {
    /// Convenience method for reading an update from a byte sequence.
    ///
    /// Uses big-endian encoding for the ID and timestamp (in seconds).
    ///
    /// # Errors
    ///
    /// Returns a [`error::DecodingError`] if the bytes cannot be decoded as a valid update.
    pub fn from_bytes<B: AsRef<[u8]>>(input: B) -> Result<Self, error::DecodingError> {
        let mut cursor = Cursor::new(input);

        io::ReadExt::read_update(&mut cursor)
    }

    /// Convenience method for getting a sequence of bytes for an update.
    ///
    /// Uses big-endian encoding for the ID and timestamp (in seconds).
    ///
    /// # Panics
    ///
    /// Panics if writing to the internal buffer fails, which should never occur in practice.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(12);

        // We are writing to a buffer, so should be fine to unwrap.
        io::WriteExt::write_update(&mut buffer, self).unwrap();

        buffer
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffValues<A> {
    pub additions: BTreeSet<A>,
    pub removals: BTreeSet<A>,
}

impl<A: Eq> Default for DiffValues<A> {
    fn default() -> Self {
        Self {
            additions: BTreeSet::new(),
            removals: BTreeSet::new(),
        }
    }
}

impl<A: Eq + Ord> DiffValues<A> {
    #[must_use]
    pub fn moved(&self) -> BTreeSet<&A> {
        self.additions.intersection(&self.removals).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.additions.len() + self.removals.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty()
    }
}

impl<A: Clone + Eq + Ord> Add<&DiffValues<A>> for &DiffValues<A> {
    type Output = DiffValues<A>;

    fn add(self, rhs: &DiffValues<A>) -> Self::Output {
        DiffValues {
            additions: self.additions.union(&rhs.additions).cloned().collect(),
            removals: self.additions.union(&rhs.removals).cloned().collect(),
        }
    }
}

impl<A> Diff<A> {
    /// Create a diff that inserts all values from scratch (empty source).
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` exceeds `i64::MAX`, which cannot occur in practice since
    /// `Vec` lengths are bounded by `isize::MAX`.
    #[must_use]
    pub fn init(values: Vec<A>) -> Self {
        let len = values.len();
        Self {
            ops: vec![Op::Insert(values)],
            // `Vec` lengths are bounded by `isize::MAX`.
            len_change: i64::try_from(len).expect("Vec length fits in i64"),
            expected_source_len: 0,
            drop_count: 0,
        }
    }

    /// Determine whether this diff makes no changes (does not check validity).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match self.ops.len() {
            1 => matches!(self.ops[0], Op::Take(_)),
            0 => true,
            _ => false,
        }
    }

    /// Verify that the expected target length is non-negative, that all operations are valid, that
    /// there are no two consecutive operations of the same type, and that the stored expected
    /// source length and drop count matches the computed values.
    #[must_use]
    pub fn validate(&self) -> bool {
        i128::from(self.expected_source_len) + i128::from(self.len_change) >= 0
            && self.ops.first().is_none_or(|first| {
                first.validate()
                    && self
                        .ops
                        .windows(2)
                        .all(|pair| pair[1].validate() && !pair[0].same_type(&pair[1]))
            })
            && {
                let (computed_expected_source_len, computed_drop_count) =
                    self.compute_expected_source_len_and_drop_count();

                self.expected_source_len == computed_expected_source_len
                    && self.drop_count == computed_drop_count
            }
    }

    /// Determine the source length that is required for this diff to be valid.
    fn compute_expected_source_len_and_drop_count(&self) -> (u64, usize) {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Take(len) => (*len, 0usize),
                Op::Drop(len) => (*len, 1usize),
                Op::Insert(_) => (0u64, 0usize),
            })
            .fold((0u64, 0usize), |(e0, d0), (e1, d1)| (e0 + e1, d0 + d1))
    }

    /// Compute the expected target length given the actual source length.
    ///
    /// # Panics
    ///
    /// Panics if `expected_source_len` exceeds `i64::MAX`, which cannot occur in practice since
    /// slice lengths are bounded by `isize::MAX`.
    ///
    /// # Errors
    ///
    /// Returns [`error::ApplicationError::UnexpectedSourceLen`] if `source_len` does not match
    /// `expected_source_len`, or [`error::ApplicationError::UnexpectedTargetLen`] if the computed
    /// target length would be negative or would not fit in `usize`.
    pub fn prepared_update_target_len(
        &self,
        source_len: usize,
    ) -> Result<usize, error::ApplicationError> {
        if source_len as u64 == self.expected_source_len {
            // `expected_source_len` is bounded by memory, so it fits in `i64` on all platforms.
            let expected_target_len = i64::try_from(self.expected_source_len)
                .expect("expected_source_len bounded by memory")
                + self.len_change;

            usize::try_from(expected_target_len).map_err(|_| {
                error::ApplicationError::UnexpectedTargetLen {
                    expected: expected_target_len,
                }
            })
        } else {
            Err(error::ApplicationError::UnexpectedSourceLen {
                expected: self.expected_source_len,
                actual: source_len as u64,
            })
        }
    }
}

impl<A: Clone> Diff<A> {
    /// Apply a diff to a source.
    ///
    /// # Panics
    ///
    /// Panics if any operation length exceeds `usize::MAX`, which cannot occur when the diff was
    /// produced from an in-memory slice.
    ///
    /// # Errors
    ///
    /// Returns an [`error::ApplicationError`] if the source length does not match, or if a
    /// `Take` or `Drop` operation extends beyond the source bounds.
    pub fn update(&self, source: &[A]) -> Result<Vec<A>, error::ApplicationError> {
        let target_len = self.prepared_update_target_len(source.len())?;
        let mut target = Vec::with_capacity(target_len);
        let mut index = 0;

        for op in &self.ops {
            match op {
                Op::Insert(values) => {
                    target.extend_from_slice(values);
                }
                Op::Take(len) => {
                    let len = usize::try_from(*len).expect("op length bounded by slice capacity");
                    if index + len > source.len() {
                        return Err(error::ApplicationError::InvalidTake(index as u64));
                    }

                    target.extend_from_slice(&source[index..index + len]);
                    index += len;
                }
                Op::Drop(len) => {
                    let len = usize::try_from(*len).expect("op length bounded by slice capacity");
                    index += len;
                    if index > source.len() {
                        return Err(error::ApplicationError::InvalidDrop((index - len) as u64));
                    }
                }
            }
        }

        if index == source.len() {
            Ok(target)
        } else {
            Err(error::ApplicationError::UnexpectedSource(index as u64))
        }
    }
}

enum Action<A> {
    Copy {
        source_index: usize,
        target_index: usize,
        len: usize,
    },
    Fill {
        target_index: usize,
        values: Vec<A>,
    },
}

impl<A: Clone + Default> Diff<A> {
    /// Apply a diff to a source in place.
    ///
    /// # Panics
    ///
    /// Panics if any operation length exceeds `usize::MAX`, which cannot occur when the diff was
    /// produced from an in-memory slice.
    ///
    /// # Errors
    ///
    /// Returns an [`error::ApplicationError`] if the source length does not match, or if a
    /// `Take` or `Drop` operation extends beyond the source bounds.
    pub fn update_in_place(self, source: &mut Vec<A>) -> Result<(), error::ApplicationError> {
        let target_len = self.prepared_update_target_len(source.len())?;

        let mut actions = Vec::with_capacity(self.ops.len() - self.drop_count);
        let mut source_index = 0;
        let mut target_index = 0;

        for op in self.ops {
            match op {
                Op::Insert(values) => {
                    let len = values.len();
                    actions.push(Action::Fill {
                        target_index,
                        values,
                    });
                    target_index += len;
                }
                Op::Take(len) => {
                    let len = usize::try_from(len).expect("op length bounded by slice capacity");
                    match source_index.cmp(&target_index) {
                        Ordering::Less => {
                            actions.push(Action::Copy {
                                source_index,
                                target_index,
                                len,
                            });
                        }
                        Ordering::Greater => actions.push(Action::Fill {
                            target_index,
                            values: source[source_index..source_index + len].to_vec(),
                        }),
                        Ordering::Equal => {}
                    }
                    source_index += len;
                    target_index += len;

                    if source_index > source.len() {
                        return Err(error::ApplicationError::InvalidTake(
                            (source_index - len) as u64,
                        ));
                    }
                }
                Op::Drop(len) => {
                    let len = usize::try_from(len).expect("op length bounded by slice capacity");
                    source_index += len;

                    if source_index > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(
                            (source_index - len) as u64,
                        ));
                    }
                }
            }
        }

        if source_index != source.len() {
            Err(error::ApplicationError::UnexpectedSource(
                source_index as u64,
            ))
        } else if target_index != target_len {
            Err(error::ApplicationError::UnexpectedTargetLen {
                expected: i64::try_from(target_len).expect("target length fits in i64"),
            })
        } else {
            if source_index < target_index {
                source.resize(target_index, Default::default());
            }

            for action in actions.into_iter().rev() {
                match action {
                    Action::Copy {
                        source_index,
                        target_index,
                        len,
                    } => {
                        let selection = source[source_index..source_index + len].to_vec();
                        source.splice(target_index..target_index + len, selection);
                    }
                    Action::Fill {
                        target_index,
                        values,
                    } => {
                        source.splice(target_index..target_index + values.len(), values);
                    }
                }
            }

            if source_index > target_index {
                source.truncate(target_index);
            }

            Ok(())
        }
    }
}

impl<A: Clone + Eq + Ord> Diff<A> {
    /// Compute the changed values for a diff applied to a source.
    ///
    /// # Panics
    ///
    /// Panics if any operation length exceeds `usize::MAX`, which cannot occur when the diff was
    /// produced from an in-memory slice.
    ///
    /// # Errors
    ///
    /// Returns an [`error::ApplicationError`] if the source length does not match, or if a
    /// `Drop` operation extends beyond the source bounds.
    pub fn values(&self, source: &[A]) -> Result<DiffValues<A>, error::ApplicationError> {
        let _target_len = self.prepared_update_target_len(source.len())?;

        let mut diff_values = DiffValues::default();
        let mut index = 0;

        for op in &self.ops {
            match op {
                Op::Insert(values) => {
                    diff_values.additions.extend(values.iter().cloned());
                }
                Op::Take(len) => {
                    index += usize::try_from(*len).expect("op length bounded by slice capacity");
                }
                Op::Drop(len) => {
                    let len = usize::try_from(*len).expect("op length bounded by slice capacity");
                    diff_values
                        .removals
                        .extend(source[index..index + len].iter().cloned());
                    index += len;
                }
            }
        }

        if index == source.len() {
            Ok(diff_values)
        } else {
            Err(error::ApplicationError::UnexpectedSource(index as u64))
        }
    }

    /// Apply a diff to a source, returning the new list and the changed values.
    ///
    /// # Panics
    ///
    /// Panics if any operation length exceeds `usize::MAX`, which cannot occur when the diff was
    /// produced from an in-memory slice.
    ///
    /// # Errors
    ///
    /// Returns an [`error::ApplicationError`] if the source length does not match, or if a
    /// `Take` or `Drop` operation extends beyond the source bounds.
    pub fn update_with_values(
        &self,
        source: &[A],
    ) -> Result<(Vec<A>, DiffValues<A>), error::ApplicationError> {
        let target_len = self.prepared_update_target_len(source.len())?;

        let mut diff_values = DiffValues::default();
        let mut target = Vec::with_capacity(target_len);
        let mut index = 0;

        for op in &self.ops {
            match op {
                Op::Insert(values) => {
                    diff_values.additions.extend(values.iter().cloned());
                    target.extend(values.clone());
                }
                Op::Take(len) => {
                    let len = usize::try_from(*len).expect("op length bounded by slice capacity");
                    if index + len > source.len() {
                        return Err(error::ApplicationError::InvalidTake(index as u64));
                    }
                    target.extend(source[index..index + len].to_vec());
                    index += len;
                }
                Op::Drop(len) => {
                    let len = usize::try_from(*len).expect("op length bounded by slice capacity");
                    if index + len > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(index as u64));
                    }
                    diff_values
                        .removals
                        .extend(source[index..index + len].iter().cloned());
                    index += len;
                }
            }
        }

        if index == source.len() {
            Ok((target, diff_values))
        } else {
            Err(error::ApplicationError::UnexpectedSource(index as u64))
        }
    }

    /// Compute the minimal diff needed to transform `source` into `target`.
    ///
    /// # Panics
    ///
    /// Panics if an internal longest increasing subsequence invariant is violated (should never
    /// occur in practice).
    ///
    /// # Errors
    ///
    /// Returns [`error::InputError::DuplicateValue`] if `target` contains duplicate elements.
    pub fn compute(source: &[A], target: &[A]) -> Result<Self, error::InputError> {
        let mut ops = Vec::with_capacity(1);
        let expected_source_len = source.len() as u64;
        let mut len_change: i64 = 0;
        let mut drop_count = 0;

        let mut source = source;
        let mut target = target;

        // Often the lists will share a long tail, so we use special handling for that case.
        let shared_tail_len = Self::count_shared_tail(source, target);

        if shared_tail_len > 0 {
            source = &source[0..source.len() - shared_tail_len];
            target = &target[0..target.len() - shared_tail_len];
        }

        let mut target_indices = BTreeMap::new();

        for (index, value) in target.iter().enumerate() {
            if let Some(previous_index) = target_indices.insert(value, index) {
                return Err(error::InputError::DuplicateValue {
                    first: previous_index,
                    second: index,
                });
            }
        }

        let source_matches = source
            .iter()
            .map(|value| target_indices.get(value))
            .collect::<Vec<_>>();

        let lis = lis::LisExt::longest_increasing_subsequence_by(
            &source_matches,
            std::cmp::Ord::cmp,
            std::option::Option::is_some,
        );

        let mut lis_remaining = lis.as_slice();

        let mut target_index = 0;

        for (source_index, (_, source_match_index)) in source.iter().zip(source_matches).enumerate()
        {
            if !lis_remaining.is_empty() && lis_remaining[0] == source_index {
                // This index was selected as a part of the LIS, so we know there's a value here.
                let source_match_index = source_match_index.unwrap();
                if *source_match_index > target_index {
                    len_change += i64::try_from(source_match_index - target_index)
                        .expect("index difference fits in i64");
                    ops.push(Op::Insert(
                        target[target_index..*source_match_index].to_vec(),
                    ));
                    target_index = *source_match_index;
                }
                Self::push_take(&mut ops, 1);
                target_index += 1;
                lis_remaining = &lis_remaining[1..];
            } else {
                drop_count += Self::push_drop(&mut ops);
                len_change -= 1;
            }
        }

        if target_index < target.len() {
            len_change +=
                i64::try_from(target.len() - target_index).expect("slice length fits in i64");
            ops.push(Op::Insert(target[target_index..].to_vec()));
        }

        Self::push_take(&mut ops, shared_tail_len);

        Ok(Self {
            ops,
            expected_source_len,
            len_change,
            drop_count,
        })
    }

    fn push_take(ops: &mut Vec<Op<A>>, len: usize) {
        if len > 0 {
            if let Some(Op::Take(old_len)) = ops.last_mut() {
                *old_len += len as u64;
            } else {
                ops.push(Op::Take(len as u64));
            }
        }
    }

    fn push_drop(ops: &mut Vec<Op<A>>) -> usize {
        if let Some(Op::Drop(old_len)) = ops.last_mut() {
            *old_len += 1u64;
            0
        } else {
            ops.push(Op::Drop(1u64));
            1
        }
    }

    fn count_shared_tail(x: &[A], y: &[A]) -> usize {
        let x_len = x.len();
        let y_len = y.len();
        let mut i = 0;

        // Clippy is wrong: we're walking from the end for both arrays.
        #[allow(clippy::suspicious_operation_groupings)]
        while x_len > i && y_len > i && x[x_len - i - 1] == y[y_len - i - 1] {
            i += 1;
        }

        i
    }
}

impl<A: Clone + Default + Eq + Ord> Diff<A> {
    /// Apply a diff to a source in place, returning the changed values.
    ///
    /// # Panics
    ///
    /// Panics if any operation length exceeds `usize::MAX`, which cannot occur when the diff was
    /// produced from an in-memory slice.
    ///
    /// # Errors
    ///
    /// Returns an [`error::ApplicationError`] if the source length does not match, or if a
    /// `Take` or `Drop` operation extends beyond the source bounds.
    pub fn update_in_place_with_values(
        self,
        source: &mut Vec<A>,
    ) -> Result<DiffValues<A>, error::ApplicationError> {
        let target_len = self.prepared_update_target_len(source.len())?;

        let mut diff_values = DiffValues::default();
        let mut actions = Vec::with_capacity(self.ops.len() - self.drop_count);
        let mut source_index = 0;
        let mut target_index = 0;

        for op in self.ops {
            match op {
                Op::Insert(values) => {
                    diff_values.additions.extend(values.iter().cloned());
                    let len = values.len();
                    actions.push(Action::Fill {
                        target_index,
                        values,
                    });
                    target_index += len;
                }
                Op::Take(len) => {
                    let len = usize::try_from(len).expect("op length bounded by slice capacity");
                    match source_index.cmp(&target_index) {
                        Ordering::Less => {
                            actions.push(Action::Copy {
                                source_index,
                                target_index,
                                len,
                            });
                        }
                        Ordering::Greater => actions.push(Action::Fill {
                            target_index,
                            values: source[source_index..source_index + len].to_vec(),
                        }),
                        Ordering::Equal => {}
                    }
                    source_index += len;
                    target_index += len;

                    if source_index > source.len() {
                        return Err(error::ApplicationError::InvalidTake(
                            (source_index - len) as u64,
                        ));
                    }
                }
                Op::Drop(len) => {
                    let len = usize::try_from(len).expect("op length bounded by slice capacity");
                    diff_values
                        .removals
                        .extend(source[source_index..source_index + len].iter().cloned());
                    source_index += len;

                    if source_index > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(
                            (source_index - len) as u64,
                        ));
                    }
                }
            }
        }

        if source_index != source.len() {
            Err(error::ApplicationError::UnexpectedSource(
                source_index as u64,
            ))
        } else if target_index != target_len {
            Err(error::ApplicationError::UnexpectedTargetLen {
                expected: i64::try_from(target_len).expect("target length fits in i64"),
            })
        } else {
            if source_index < target_index {
                source.resize(target_index, Default::default());
            }

            for action in actions.into_iter().rev() {
                match action {
                    Action::Copy {
                        source_index,
                        target_index,
                        len,
                    } => {
                        let selection = source[source_index..source_index + len].to_vec();
                        source.splice(target_index..target_index + len, selection);
                    }
                    Action::Fill {
                        target_index,
                        values,
                    } => {
                        source.splice(target_index..target_index + values.len(), values);
                    }
                }
            }

            if source_index > target_index {
                source.truncate(target_index);
            }

            Ok(diff_values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Diff, Op, Update,
        io::{ReadExt, WriteExt},
    };
    use crate::diff::DiffValues;
    use chrono::SubsecRound;
    use quickcheck::Arbitrary;
    use rand::{rng, seq::SliceRandom};
    use std::collections::BTreeSet;

    impl<A: Arbitrary> Arbitrary for Op<A> {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            match g.choose(&[-1, 0, 1]).unwrap() {
                0 => {
                    let mut values: Vec<A> = Arbitrary::arbitrary(g);

                    if values.is_empty() {
                        values.push(Arbitrary::arbitrary(g));
                    }

                    Self::Insert(values)
                }
                other => {
                    let count: u8 = Arbitrary::arbitrary(g);
                    let count = if count == 0 { 1u64 } else { count as u64 };

                    if *other == -1 {
                        Self::Drop(count)
                    } else {
                        Self::Take(count)
                    }
                }
            }
        }
    }

    impl<A: Arbitrary + Clone> Arbitrary for Diff<A> {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let ops: Vec<Op<A>> = Arbitrary::arbitrary(g);
            let mut clean_ops = vec![];
            let mut expected_source_len: u64 = 0;
            let mut len_change: i64 = 0;
            let mut drop_count = 0;

            for op in ops {
                match &op {
                    Op::Take(count) => {
                        expected_source_len += count;
                    }
                    Op::Drop(count) => {
                        expected_source_len += count;
                        len_change -= *count as i64;
                        drop_count += 1;
                    }
                    Op::Insert(values) => {
                        len_change += values.len() as i64;
                    }
                }

                match clean_ops.last_mut() {
                    Some(last) => match (last, &op) {
                        (Op::Take(last_count), Op::Take(this_count)) => {
                            *last_count += this_count;
                        }
                        (Op::Drop(last_count), Op::Drop(this_count)) => {
                            *last_count += this_count;
                            drop_count -= 1;
                        }
                        (Op::Insert(last_values), Op::Insert(this_values)) => {
                            last_values.extend_from_slice(this_values);
                        }
                        (_, _) => {
                            clean_ops.push(op);
                        }
                    },
                    None => {
                        clean_ops.push(op);
                    }
                }
            }

            Self {
                ops: clean_ops,
                expected_source_len,
                len_change,
                drop_count,
            }
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct DistinctVec<A>(Vec<A>);

    impl<A> AsRef<[A]> for DistinctVec<A> {
        fn as_ref(&self) -> &[A] {
            &self.0
        }
    }

    impl<A: Arbitrary + Clone + Eq + Ord + 'static> Arbitrary for DistinctVec<A> {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let base = BTreeSet::<A>::arbitrary(g);
            let mut output = base.into_iter().collect::<Vec<_>>();
            output.shuffle(&mut rng());

            Self(output)
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct SortedDistinctVec<A>(Vec<A>);

    impl<A> AsRef<[A]> for SortedDistinctVec<A> {
        fn as_ref(&self) -> &[A] {
            &self.0
        }
    }

    impl<A: Arbitrary + Clone + Ord + Ord + 'static> Arbitrary for SortedDistinctVec<A> {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let base = BTreeSet::<A>::arbitrary(g);
            let mut output = base.into_iter().collect::<Vec<_>>();
            output.sort();

            Self(output)
        }
    }

    #[quickcheck_macros::quickcheck]
    fn update(source: DistinctVec<u8>, diff: Diff<u8>) -> bool {
        diff.validate()
            && {
                let result = diff.update(source.as_ref());

                if diff.expected_source_len == source.as_ref().len() as u64 {
                    source.as_ref().len() as i64 + diff.len_change == result.unwrap().len() as i64
                } else {
                    matches!(
                        result,
                        Err(
                            super::error::ApplicationError::UnexpectedSourceLen { expected, actual },
                        ) if expected == diff.expected_source_len && actual == source.as_ref().len() as u64
                    )
                }
            }
    }

    #[quickcheck_macros::quickcheck]
    fn update_in_place_init(source: DistinctVec<u64>) -> bool {
        let mut new_source = vec![];
        let diff = Diff::init(source.as_ref().to_vec());

        diff.update_in_place(&mut new_source).unwrap();

        new_source == source.as_ref()
    }

    #[quickcheck_macros::quickcheck]
    fn is_identity(source: DistinctVec<u64>) -> bool {
        let diff = Diff::compute(source.as_ref(), source.as_ref()).unwrap();

        diff.is_identity()
    }

    #[quickcheck_macros::quickcheck]
    fn round_trip_distinct(source: DistinctVec<u64>, target: DistinctVec<u64>) -> bool {
        let diff = Diff::compute(source.as_ref(), target.as_ref()).unwrap();
        let is_valid = diff.validate() && diff.expected_source_len == source.as_ref().len() as u64;

        let len_change = diff.len_change;
        let expected_len_change = target.as_ref().len() as i64 - source.as_ref().len() as i64;

        let computed_target = diff.update(source.as_ref()).unwrap();

        let mut source_copy = source.as_ref().to_vec();
        diff.update_in_place(&mut source_copy).unwrap();

        is_valid
            && computed_target == target.as_ref()
            && source_copy == target.as_ref()
            && len_change == expected_len_change
    }

    #[quickcheck_macros::quickcheck]
    fn round_trip_sorted_distinct(
        source: SortedDistinctVec<u64>,
        target: SortedDistinctVec<u64>,
    ) -> bool {
        let diff = Diff::compute(source.as_ref(), target.as_ref()).unwrap();
        let is_valid = diff.validate() && diff.expected_source_len == source.as_ref().len() as u64;

        let len_change = diff.len_change;
        let expected_len_change = target.as_ref().len() as i64 - source.as_ref().len() as i64;

        let computed_target = diff.update(source.as_ref()).unwrap();

        let mut source_copy = source.as_ref().to_vec();
        diff.update_in_place(&mut source_copy).unwrap();

        is_valid
            && computed_target == target.as_ref()
            && source_copy == target.as_ref()
            && len_change == expected_len_change
    }

    #[quickcheck_macros::quickcheck]
    fn round_trip_distinct_via_bytes(source: DistinctVec<u64>, target: DistinctVec<u64>) -> bool {
        let diff = Diff::compute(source.as_ref(), target.as_ref()).unwrap();
        let is_valid = diff.validate() && diff.expected_source_len == source.as_ref().len() as u64;

        let len_change = diff.len_change;
        let expected_len_change = target.as_ref().len() as i64 - source.as_ref().len() as i64;

        let mut buffer = vec![];
        buffer.write_diff(&diff).unwrap();

        let new_diff: Diff<u64> = buffer.as_slice().read_diff().unwrap();
        let computed_target = new_diff.update(source.as_ref()).unwrap();

        let mut source_copy = source.as_ref().to_vec();
        new_diff.clone().update_in_place(&mut source_copy).unwrap();

        is_valid
            && new_diff == diff
            && computed_target == target.as_ref()
            && source_copy == target.as_ref()
            && len_change == expected_len_change
    }

    #[quickcheck_macros::quickcheck]
    fn round_trip_sorted_distinct_via_bytes(
        source: SortedDistinctVec<u64>,
        target: SortedDistinctVec<u64>,
    ) -> bool {
        let diff = Diff::compute(source.as_ref(), target.as_ref()).unwrap();
        let is_valid = diff.validate() && diff.expected_source_len == source.as_ref().len() as u64;

        let len_change = diff.len_change;
        let expected_len_change = target.as_ref().len() as i64 - source.as_ref().len() as i64;

        let mut buffer = vec![];
        buffer.write_diff(&diff).unwrap();

        let new_diff: Diff<u64> = buffer.as_slice().read_diff().unwrap();
        let computed_target = new_diff.update(source.as_ref()).unwrap();

        let mut source_copy = source.as_ref().to_vec();
        new_diff.clone().update_in_place(&mut source_copy).unwrap();

        is_valid
            && new_diff == diff
            && computed_target == target.as_ref()
            && source_copy == target.as_ref()
            && len_change == expected_len_change
    }

    #[quickcheck_macros::quickcheck]
    fn round_trip_distinct_updates_via_bytes(
        id: u64,
        source: DistinctVec<u64>,
        target: DistinctVec<u64>,
    ) -> bool {
        let diff = Diff::compute(source.as_ref(), target.as_ref()).unwrap();
        let is_valid = diff.validate() && diff.expected_source_len == source.as_ref().len() as u64;

        let timestamp = chrono::Utc::now().trunc_subsecs(0);
        let update = Update::new(timestamp, id, diff);

        let mut buffer = vec![];
        buffer.write_update(&update).unwrap();

        let read_update: Update<u64> = buffer.as_slice().read_update().unwrap();

        is_valid && read_update == update
    }

    #[test]
    fn test_simple_example() {
        let source = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"];
        let target = vec!["X", "C", "D", "M", "F", "G", "H", "I", "P", "Q", "L"];
        let diff = Diff::compute(&source, &target).unwrap();

        assert!(diff.validate());
        assert_eq!(diff.expected_source_len, source.len() as u64);

        let expected_diff = Diff {
            ops: vec![
                Op::Drop(2),
                Op::Insert(vec!["X"]),
                Op::Take(2),
                Op::Drop(1),
                Op::Insert(vec!["M"]),
                Op::Take(4),
                Op::Drop(2),
                Op::Insert(vec!["P", "Q"]),
                Op::Take(1),
            ],
            expected_source_len: source.len() as u64,
            len_change: -1,
            drop_count: 3,
        };

        assert_eq!(diff, expected_diff);

        let expected_diff_values = DiffValues {
            additions: BTreeSet::from_iter(["X", "M", "P", "Q"]),
            removals: BTreeSet::from_iter(["A", "B", "E", "J", "K"]),
        };

        let diff_values = diff.values(&source).unwrap();
        assert_eq!(diff_values, expected_diff_values);

        let b_computed = diff.update(&source).unwrap();
        assert_eq!(b_computed, target);

        let (b_computed, diff_values) = diff.update_with_values(&source).unwrap();
        assert_eq!(b_computed, target);
        assert_eq!(diff_values, expected_diff_values);

        let mut source_copy = source.clone();
        diff.clone().update_in_place(&mut source_copy).unwrap();
        assert_eq!(source_copy, target);

        let mut source_copy = source.clone();
        let diff_values = diff
            .clone()
            .update_in_place_with_values(&mut source_copy)
            .unwrap();
        assert_eq!(source_copy, target);
        assert_eq!(diff_values, expected_diff_values);
    }

    #[test]
    fn test_compute_large() {
        let mut source = (0..100_000_000).collect::<Vec<_>>();
        let mut target = (0..100_000_000).collect::<Vec<_>>();
        source[999997] = 100_000_001;
        target[0] = 100_000_002;

        let diff = Diff::compute(&source, &target).unwrap();

        assert!(diff.validate());
        assert_eq!(diff.expected_source_len, source.len() as u64);

        let computed_target = diff.update(&source).unwrap();

        assert_eq!(computed_target, target);
    }

    #[test]
    fn test_count_shared_tail() {
        let x = &[1, 2, 3, 4, 5];
        let y = &[0, 1, 9, 3, 4, 5];

        assert_eq!(Diff::count_shared_tail(x, y), 3);

        let x = &[1, 2, 3, 4, 5, 9];
        let y = &[0, 1, 9, 3, 4, 5];

        assert_eq!(Diff::count_shared_tail(x, y), 0);

        let x = &[0, 1, 9, 3, 4, 5];
        let y = &[0, 1, 9, 3, 4, 5];

        assert_eq!(Diff::count_shared_tail(x, y), 6);

        let x = &[0, 1, 9, 3, 4, 5];
        let y = &[7, 0, 1, 9, 3, 4, 5];

        assert_eq!(Diff::count_shared_tail(x, y), 6);

        let x = &[7, 0, 1, 9, 3, 4, 5];
        let y = &[0, 1, 9, 3, 4, 5];

        assert_eq!(Diff::count_shared_tail(x, y), 6);
    }

    #[test]
    fn test_real_world_example() {
        let example_1 = id_lines(std::include_str!(
            "../../examples/1538705932799987712-1692693920.txt"
        ));
        let example_2 = id_lines(std::include_str!(
            "../../examples/1538705932799987712-1692821177.txt"
        ));

        let diff_1_bytes = std::include_bytes!(
            "../../examples/1538705932799987712-1692693920-1692821177.diff.bin"
        );

        let diff_1 = Diff::compute(&example_1, &example_2).unwrap();

        assert_eq!(diff_1.len_change, 567);
        assert_eq!(diff_1.ops.len(), 404);

        let mut written_diff_1_bytes = vec![];
        written_diff_1_bytes.write_diff(&diff_1).unwrap();

        assert_eq!(written_diff_1_bytes, diff_1_bytes);

        let read_diff_1 = diff_1_bytes.as_slice().read_diff().unwrap();

        assert_eq!(read_diff_1, diff_1);
    }

    fn id_lines(input: &str) -> Vec<u64> {
        input
            .split('\n')
            .filter(|line| !line.is_empty())
            .map(|line| line.parse::<u64>().unwrap())
            .collect()
    }
}
