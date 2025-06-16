from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.paillier_encryption.paillier_context import \
    PaillierContext

if __name__ == '__main__':
    paillier_he = PaillierContext()
    my_key = paillier_he.get_key_pair()

    g, n = my_key.public_key
    m1 = 71
    m2 = 29
    r1 = paillier_he.get_r_for_encryption(n)
    r2 = paillier_he.get_r_for_encryption(n)

    c1 = paillier_he.encrypt(m1, my_key.public_key, r1)
    c2 = paillier_he.encrypt(m2, my_key.public_key, r2)

    # validate simple encryption works
    print(paillier_he.decrypt(c1) == m1)

    # validate that d(e(m1 + m2)) == d(e(m1) * e(m2))
    en_mult = (c1 * c2) % (n * n)
    add_en = paillier_he.encrypt(m1 + m2, my_key.public_key, r1*r2)

    print(paillier_he.decrypt(en_mult) == paillier_he.decrypt(add_en))

    # validate that d(e(m1)^m2) == m1m2 mod n
    en_pow = pow(c1, m2, n * n)
    mul_en = (m2 * m1) % n

    print(paillier_he.decrypt(en_pow) == mul_en)

    # validate that d(e(m1)^k) == k * m1 mod n
    k = 23
    en_pow = pow(c1, k, n * n)
    mul_en = (k * m1) % n

    print(paillier_he.decrypt(en_pow) == mul_en)
    print("Success")