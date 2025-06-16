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
    en_mult = (c1 * c2) % (n * n)
    add_en = paillier_he.encrypt(m1 + m2, my_key.public_key, r1*r2)

    print(paillier_he.decrypt(en_mult) == paillier_he.decrypt(add_en))
    print("Success")