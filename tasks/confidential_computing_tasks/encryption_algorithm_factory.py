from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.light_phe_security_algorithm import \
    LightPHESecurityAlgorithm, LightPHEAlgorithms
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.paillier_security_algorithm import \
    PaillierSecurityAlgorithm
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.rsa_security_algorithm import \
    RSASecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL


class EncryptionAlgorithmFactory:
    @staticmethod
    def create_security_algorithm(encryption_algorithm: EncryptionType, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL) -> SecurityAlgorithm:
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
        else:
            raise ValueError("Unknown encryption algorithm")