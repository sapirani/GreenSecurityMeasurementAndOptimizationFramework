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
    FernetAES = 15
    PycryptoAES = 16
    PycryptoDES = 17
    PycryptoBlowfish = 18
    PycryptoChaCha20 = 19
    PycryptoArc4 = 20
    PycryptoRSA = 21


class EncryptionMode(Enum):
    ECB = 1
    CBC = 2
    CFB = 3
    OFB = 4
    CTR = 5
    GCM = 6
    OCB = 7
    OPENPGP = 8
    CCM = 9
    EAX = 10
    SIV = 11
