from typing import Optional

from tenseal import CKKSVector, BFVVector

from tasks.confidential_computing_tasks.abstract_security_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.asymmetric_encryption_tasks.pycryptodome_asym_security_algorithm import \
    PycryptodomeAsymmetricSecurityAlgorithm, PycryptodomeAsymmetricAlgorithms
from tasks.confidential_computing_tasks.encryption_type import EncryptionType, EncryptionMode
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.light_phe_security_algorithm import \
    LightPHESecurityAlgorithm, LightPHEAlgorithms
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.paillier_security_algorithm import \
    PaillierSecurityAlgorithm
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.rsa_security_algorithm import \
    RSASecurityAlgorithm
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.tenseal_he_security_algorithm import \
    TensealSecurityAlgorithm, TensealSchemas
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL
from tasks.confidential_computing_tasks.symmetric_encryption_tasks.fernet_aes_security_algorithm import \
    FernetAESSecurityAlgorithm
from tasks.confidential_computing_tasks.symmetric_encryption_tasks.pycryptodome_sym_security_algorithm import \
    PycryptodomeSymmetricSecurityAlgorithm, PycryptodomeSymmetricAlgorithms


class EncryptionAlgorithmFactory:
    @staticmethod
    def create_security_algorithm(encryption_algorithm: EncryptionType, cipher_block_mode: Optional[str],
                                  min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL) \
                                  -> SecurityAlgorithm:
        if encryption_algorithm == EncryptionType.Paillier:
            return PaillierSecurityAlgorithm(min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.RSA:
            return RSASecurityAlgorithm(min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheRSA:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.RSA, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheNaccacheStern:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.NACCACHE_STERN, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheGoldwasserMicali:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.GOLDWASSER_MICALI, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheEllipticCurveElGamal:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.ELLIPTICCURVE_ELGAMAL, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPhePaillier:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.PAILLIER, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheBenaloh:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.BENALOH, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheElGamal:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.ELGAMAL, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheOkamotoUchiyama:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.OKAMOTO_UCHIYAMA, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheDamgardJurik:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.DAMGARD_JURIK, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.LightPheExponentialElGamal:
            return LightPHESecurityAlgorithm(LightPHEAlgorithms.EXPONENTIAL_ELGAMAL, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.CKKSTenseal:
            return TensealSecurityAlgorithm[CKKSVector](TensealSchemas.CKKS, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.BFVTenseal:
            return TensealSecurityAlgorithm[BFVVector](TensealSchemas.BFV, min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.FernetAES:
            return FernetAESSecurityAlgorithm(min_key_val, max_key_val)
        elif encryption_algorithm == EncryptionType.PycryptoAES:
            return PycryptodomeSymmetricSecurityAlgorithm(algorithm=PycryptodomeSymmetricAlgorithms.AES,
                                                          mode=cipher_block_mode)
        elif encryption_algorithm == EncryptionType.PycryptoDES:
            return PycryptodomeSymmetricSecurityAlgorithm(algorithm=PycryptodomeSymmetricAlgorithms.DES,
                                                          mode=cipher_block_mode)
        elif encryption_algorithm == EncryptionType.PycryptoBlowfish:
            return PycryptodomeSymmetricSecurityAlgorithm(algorithm=PycryptodomeSymmetricAlgorithms.BLOWFISH,
                                                          mode=cipher_block_mode)
        elif encryption_algorithm == EncryptionType.PycryptoArc4:
            return PycryptodomeSymmetricSecurityAlgorithm(algorithm=PycryptodomeSymmetricAlgorithms.ARC4,
                                                          mode=cipher_block_mode)
        elif encryption_algorithm == EncryptionType.PycryptoChaCha20:
            return PycryptodomeSymmetricSecurityAlgorithm(algorithm=PycryptodomeSymmetricAlgorithms.CHACHA20,
                                                          mode=cipher_block_mode)
        elif encryption_algorithm == EncryptionType.PycryptoRSA:
            return PycryptodomeAsymmetricSecurityAlgorithm(algorithm=PycryptodomeAsymmetricAlgorithms.RSA)
        else:
            raise Exception('Encryption algorithm not supported')
