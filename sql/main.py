from sql.v1 import loadv1


def main():
    # sqlite3.register_adapter(np.int64, lambda val: int(val))
    # sqlite3.register_adapter(np.float64, lambda val: float(val))
    #
    # engine, session = connect(database='herbario_resultados')
    # base = get_base()
    # base.metadata.create_all(engine)

    # loadv2(session)
    loadv1(None)

    # close(engine, session)


if __name__ == '__main__':
    main()
