use std::marker::PhantomData;

use arithmetic::field::Field;
use util::fiat_shamir::{Proof, Transcript};

use crate::{CommitmentSerde, PolyCommitProver, PolyCommitVerifier};

#[derive(Debug, Clone, Default)]
pub struct NilCommitment<F: Field>(PhantomData<F>);

impl<F: Field> CommitmentSerde for NilCommitment<F> {
    fn size(_nv: usize, _np: usize) -> usize {
        0
    }

    fn serialize_into(&self, _buffer: &mut [u8]) {}

    fn deserialize_from(_proof: &mut Proof, _var_num: usize, _poly_num: usize) -> Self {
        NilCommitment::default()
    }
}

#[derive(Debug, Clone)]
pub struct NilPcProver<F: Field> {
    _evals: Vec<Vec<F::BaseField>>,
}

impl<F: Field> PolyCommitProver<F> for NilPcProver<F> {
    type Param = ();
    type Commitment = NilCommitment<F>;

    fn new(_pp: &(), evals: &[Vec<F::BaseField>]) -> Self {
        NilPcProver {
            _evals: evals.iter().map(|x| x.clone()).collect(),
        }
    }

    fn commit(&self) -> Self::Commitment {
        NilCommitment::default()
    }

    fn open(
        _pp: &Self::Param,
        _provers: Vec<&Self>,
        _point: Vec<F>,
        _transcript: &mut Transcript,
    ) {
    }
}

#[derive(Debug, Clone)]
pub struct NilPcVerifier<F: Field> {
    _commit: NilCommitment<F>,
}

impl<F: Field> PolyCommitVerifier<F> for NilPcVerifier<F> {
    type Param = ();
    type Commitment = NilCommitment<F>;

    fn new(_pp: &Self::Param, commit: Self::Commitment, _poly_num: usize) -> Self {
        NilPcVerifier { _commit: commit }
    }

    fn verify(
        _pp: &Self::Param,
        _verifiers: Vec<&Self>,
        _point: Vec<F>,
        _evals: Vec<Vec<F>>,
        _transcript: &mut Transcript,
        _proof: &mut Proof,
    ) -> bool {
        true
    }
}
