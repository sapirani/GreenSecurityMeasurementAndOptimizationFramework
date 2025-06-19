from enum import Enum

class EncryptionType(Enum):
    Paillier = 1
    RSA = 2
    LightPheRSA = 3
    LightPheElGamal = 4
    LightPheExponentialElGamal = 5
    LightPhePaillier = 6
    LightPheDamgardJurik = 7
    LightPheOkamotoUchiyama = 8
    LightPheBenaloh = 9
    LightPheNaccacheStern = 10
    LightPheGoldwasserMicali = 11
    LightPheEllipticCurveElGamal = 12
    CKKSTenseal = 13
    BFVTenseal = 14