use byteorder::{BigEndian, ReadBytesExt};
use chrono::{DateTime, TimeZone, Utc};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::io::Cursor;
use std::ops::Add;

pub mod error;
pub mod io;

/// A single transformation operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op<A> {
    Take(usize),
    Drop(usize),
    Insert(Vec<A>),
}

/// The operations necessary to transform one list of distinct elements into another.
///
/// The diff implementation is optimized for the case where the order of elements is mostly
/// unchanged (Twitter follower lists, for example), but will produce correct results in the
/// general case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diff<A> {
    pub ops: Vec<Op<A>>,
    pub len_change: isize,
    drop_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Update<A> {
    pub id: A,
    pub timestamp: DateTime<Utc>,
    pub diff: Diff<A>,
}

impl<A> Update<A> {
    pub fn new(id: A, timestamp: DateTime<Utc>, diff: Diff<A>) -> Self {
        Self {
            id,
            timestamp,
            diff,
        }
    }
}

impl Update<u64> {
    /// Convenience method for getting a sequence of bytes for an update.
    ///
    /// Uses big-endian encoding for the ID and timestamp (in seconds). The standard diff encoding is used,
    /// except that an empty diff is represented by zero bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(12);
        buffer.extend_from_slice(&self.id.to_be_bytes());
        buffer.extend_from_slice(&(self.timestamp.timestamp() as u32).to_be_bytes());
        // We are writing to a buffer, so should be fine to unwrap.
        io::write_diff(&mut buffer, &self.diff).unwrap();
        buffer
    }

    pub fn from_bytes<B: AsRef<[u8]>>(input: B) -> Result<Self, error::DecodingError> {
        let mut cursor = Cursor::new(input);

        let id = cursor
            .read_u64::<BigEndian>()
            .map_err(|_| error::DecodingError::InvalidIndex(cursor.position()))?;

        let timestamp = cursor
            .read_u32::<BigEndian>()
            .ok()
            .and_then(|timestamp_s| Utc.timestamp_opt(timestamp_s as i64, 0).single())
            .ok_or_else(|| error::DecodingError::InvalidTimestamp(cursor.position()))?;

        let diff = io::read_diff(cursor)?;

        Ok(Self {
            id,
            timestamp,
            diff,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffValues<A: Eq + Hash> {
    pub additions: HashSet<A>,
    pub removals: HashSet<A>,
}

impl<A: Eq + Hash> Default for DiffValues<A> {
    fn default() -> Self {
        Self {
            additions: HashSet::new(),
            removals: HashSet::new(),
        }
    }
}

impl<A: Eq + Hash> DiffValues<A> {
    pub fn moved(&self) -> HashSet<&A> {
        self.additions.intersection(&self.removals).collect()
    }

    pub fn len(&self) -> usize {
        self.additions.len() + self.removals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty()
    }
}

impl<A: Clone + Eq + Hash> Add<&DiffValues<A>> for &DiffValues<A> {
    type Output = DiffValues<A>;

    fn add(self, rhs: &DiffValues<A>) -> Self::Output {
        DiffValues {
            additions: self.additions.union(&rhs.additions).cloned().collect(),
            removals: self.additions.union(&rhs.removals).cloned().collect(),
        }
    }
}

impl<A: Clone> Diff<A> {
    pub fn new(values: Vec<A>) -> Self {
        let len = values.len();
        Self {
            ops: vec![Op::Insert(values)],
            len_change: len as isize,
            drop_count: 0,
        }
    }

    pub fn update(&self, source: &[A]) -> Result<Vec<A>, error::ApplicationError> {
        let target_len = (source.len() as isize + self.len_change) as usize;
        let mut target = Vec::with_capacity(target_len);
        let mut index = 0;

        for op in &self.ops {
            match op {
                Op::Insert(values) => {
                    target.extend(values.clone());
                }
                Op::Take(len) => {
                    target.extend(source[index..index + len].to_vec());
                    index += len;
                    if index > source.len() {
                        return Err(error::ApplicationError::InvalidTake(index - len));
                    }
                }
                Op::Drop(len) => {
                    index += len;
                    if index > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(index - len));
                    }
                }
            }
        }

        if index == source.len() {
            Ok(target)
        } else {
            Err(error::ApplicationError::UnexpectedData(index))
        }
    }
}

impl<A> Diff<A> {
    /// Determine whether this diff does nothing.
    ///
    /// If a source length is provided, checks whether the diff is valid for this source.
    pub fn is_empty(&self, source_len: Option<usize>) -> bool {
        self.ops.len() == 1
            && matches!(self.ops[0], Op::Take(len) if source_len.filter(|source_len| *source_len != len).is_none())
    }

    /// Determine the source length that is required for this diff to be valid.
    pub fn expected_source_len(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Take(len) => *len,
                Op::Drop(len) => *len,
                Op::Insert(_) => 0,
            })
            .sum()
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
    pub fn update_in_place(self, source: &mut Vec<A>) -> Result<(), error::ApplicationError> {
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
                        return Err(error::ApplicationError::InvalidTake(source_index - len));
                    }
                }
                Op::Drop(len) => {
                    source_index += len;

                    if source_index > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(source_index - len));
                    }
                }
            }
        }

        if source_index != source.len() {
            Err(error::ApplicationError::UnexpectedData(source_index))
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

impl<A: Clone + Eq + Hash> Diff<A> {
    pub fn values(&self, source: &[A]) -> Result<DiffValues<A>, error::ApplicationError> {
        let mut diff_values = DiffValues::default();
        let mut index = 0;

        for op in &self.ops {
            match op {
                Op::Insert(values) => {
                    diff_values.additions.extend(values.iter().cloned());
                }
                Op::Take(len) => {
                    index += len;
                }
                Op::Drop(len) => {
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
            Err(error::ApplicationError::UnexpectedData(index))
        }
    }

    pub fn update_with_values(
        &self,
        source: &[A],
    ) -> Result<(Vec<A>, DiffValues<A>), error::ApplicationError> {
        let mut diff_values = DiffValues::default();
        let target_len = (source.len() as isize + self.len_change) as usize;
        let mut target = Vec::with_capacity(target_len);
        let mut index = 0;

        for op in &self.ops {
            match op {
                Op::Insert(values) => {
                    diff_values.additions.extend(values.iter().cloned());
                    target.extend(values.clone());
                }
                Op::Take(len) => {
                    if index + len > source.len() {
                        return Err(error::ApplicationError::InvalidTake(index));
                    }
                    target.extend(source[index..index + len].to_vec());
                    index += len;
                }
                Op::Drop(len) => {
                    if index + len > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(index));
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
            Err(error::ApplicationError::UnexpectedData(index))
        }
    }

    pub fn compute(source: &[A], target: &[A]) -> Result<Self, error::InputError> {
        let mut ops = Vec::with_capacity(1);
        let mut drop_count = 0;
        let mut len_change: isize = 0;

        let mut source = source;
        let mut target = target;

        // Often the lists will share a long tail, so we use special handling for that case.
        let shared_tail_len = Self::count_shared_tail(source, target);

        if shared_tail_len > 0 {
            source = &source[0..source.len() - shared_tail_len];
            target = &target[0..target.len() - shared_tail_len];
        }

        let mut target_indices = HashMap::new();

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
            |a, b| a.cmp(b),
            |value| value.is_some(),
        );

        let mut lis_remaining = lis.as_slice();

        let mut target_index = 0;

        for (source_index, (_, source_match_index)) in source.iter().zip(source_matches).enumerate()
        {
            if !lis_remaining.is_empty() && lis_remaining[0] == source_index {
                // This index was selected as a part of the LIS, so we know there's a value here.
                let source_match_index = source_match_index.unwrap();
                if *source_match_index > target_index {
                    len_change += (source_match_index - target_index) as isize;
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
            len_change += (target.len() - target_index) as isize;
            ops.push(Op::Insert(target[target_index..].to_vec()));
        }

        Self::push_take(&mut ops, shared_tail_len);

        Ok(Self {
            ops,
            len_change,
            drop_count,
        })
    }

    fn push_take(ops: &mut Vec<Op<A>>, len: usize) {
        if len > 0 {
            if let Some(Op::Take(old_len)) = ops.last_mut() {
                *old_len += len;
            } else {
                ops.push(Op::Take(len))
            }
        }
    }

    fn push_drop(ops: &mut Vec<Op<A>>) -> usize {
        if let Some(Op::Drop(old_len)) = ops.last_mut() {
            *old_len += 1;
            0
        } else {
            ops.push(Op::Drop(1));
            1
        }
    }

    fn count_shared_tail(x: &[A], y: &[A]) -> usize {
        let x_len = x.len();
        let y_len = y.len();
        let mut i = 0;

        while x_len > i && y_len > i && x[x_len - i - 1] == y[y_len - i - 1] {
            i += 1;
        }

        i
    }
}

impl<A: Clone + Eq + Hash + Default> Diff<A> {
    pub fn update_in_place_with_values(
        self,
        source: &mut Vec<A>,
    ) -> Result<DiffValues<A>, error::ApplicationError> {
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
                        return Err(error::ApplicationError::InvalidTake(source_index - len));
                    }
                }
                Op::Drop(len) => {
                    diff_values
                        .removals
                        .extend(source[source_index..source_index + len].iter().cloned());
                    source_index += len;

                    if source_index > source.len() {
                        return Err(error::ApplicationError::InvalidDrop(source_index - len));
                    }
                }
            }
        }

        if source_index != source.len() {
            Err(error::ApplicationError::UnexpectedData(source_index))
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
    use quickcheck::{Arbitrary, quickcheck};
    use rand::{rng, seq::SliceRandom};
    use std::collections::HashSet;
    use std::hash::Hash;

    use crate::diff::DiffValues;

    use super::{
        Diff, Op,
        io::{read_diff, write_diff},
    };

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct DistinctVec<A>(Vec<A>);

    impl<A: Arbitrary + Clone + Eq + Hash + 'static> Arbitrary for DistinctVec<A> {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let base = HashSet::<A>::arbitrary(g);
            let mut output = base.into_iter().collect::<Vec<_>>();
            output.shuffle(&mut rng());

            Self(output)
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct SortedDistinctVec<A>(Vec<A>);

    impl<A: Arbitrary + Clone + Ord + Hash + 'static> Arbitrary for SortedDistinctVec<A> {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let base = HashSet::<A>::arbitrary(g);
            let mut output = base.into_iter().collect::<Vec<_>>();
            output.sort();

            Self(output)
        }
    }

    quickcheck! {
        fn update_in_place_new(source: DistinctVec<u64>) -> bool {
            let mut new_source = vec![];
            let diff = Diff::new(source.0.clone());

            diff.update_in_place(&mut new_source).unwrap();

            new_source == source.0
        }
    }

    quickcheck! {
        fn round_trip_distinct(source: DistinctVec<u64>, target: DistinctVec<u64>) -> bool {
            let diff = Diff::compute(&source.0, &target.0).unwrap();

            let len_change = diff.len_change;
            let expected_len_change = target.0.len() as isize - source.0.len() as isize;

            let computed_target = diff.update(&source.0).unwrap();

            let mut source_copy = source.0.clone();
            diff.update_in_place(&mut source_copy).unwrap();

            computed_target == target.0 && source_copy == target.0 && len_change == expected_len_change
        }
    }

    quickcheck! {
        fn round_trip_sorted_distinct(source: SortedDistinctVec<u64>, target: SortedDistinctVec<u64>) -> bool {
            let diff = Diff::compute(&source.0, &target.0).unwrap();

            let len_change = diff.len_change;
            let expected_len_change = target.0.len() as isize - source.0.len() as isize;

            let computed_target = diff.update(&source.0).unwrap();

            let mut source_copy = source.0.clone();
            diff.update_in_place(&mut source_copy).unwrap();

            computed_target == target.0 && source_copy == target.0 && len_change == expected_len_change
        }
    }

    quickcheck! {
        fn round_trip_distinct_via_bytes(source: DistinctVec<u64>, target: DistinctVec<u64>) -> bool {
            let diff = Diff::compute(&source.0, &target.0).unwrap();

            let len_change = diff.len_change;
            let expected_len_change = target.0.len() as isize - source.0.len() as isize;

            let mut buffer = vec![];
            write_diff(&diff, &mut buffer).unwrap();

            let new_diff: Diff<u64> = read_diff(buffer.as_slice()).unwrap();
            let computed_target = new_diff.update(&source.0).unwrap();

            let mut source_copy = source.0.clone();
            new_diff.clone().update_in_place(&mut source_copy).unwrap();

            new_diff == diff && computed_target == target.0 && source_copy == target.0 && len_change == expected_len_change
        }
    }

    quickcheck! {
        fn round_trip_sorted_distinct_via_bytes(source: SortedDistinctVec<u64>, target: SortedDistinctVec<u64>) -> bool {
            let diff = Diff::compute(&source.0, &target.0).unwrap();

            let len_change = diff.len_change;
            let expected_len_change = target.0.len() as isize - source.0.len() as isize;

            let mut buffer = vec![];
            write_diff(&diff, &mut buffer).unwrap();

            let new_diff: Diff<u64> = read_diff(buffer.as_slice()).unwrap();
            let computed_target = new_diff.update(&source.0).unwrap();

            let mut source_copy = source.0.clone();
            new_diff.clone().update_in_place(&mut source_copy).unwrap();

            new_diff == diff && computed_target == target.0 && source_copy == target.0 && len_change == expected_len_change
        }
    }

    #[test]
    fn test_simple_example() {
        let source = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"];
        let target = vec!["X", "C", "D", "M", "F", "G", "H", "I", "P", "Q", "L"];
        let diff = Diff::compute(&source, &target).unwrap();

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
            drop_count: 3,
            len_change: -1,
        };
        assert_eq!(diff, expected_diff);

        let expected_diff_values = DiffValues {
            additions: HashSet::from_iter(["X", "M", "P", "Q"]),
            removals: HashSet::from_iter(["A", "B", "E", "J", "K"]),
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
        write_diff(&diff_1, &mut written_diff_1_bytes).unwrap();

        assert_eq!(written_diff_1_bytes, diff_1_bytes);

        let read_diff_1 = read_diff(diff_1_bytes).unwrap();

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
