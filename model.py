import sqlalchemy as sa
import sqlalchemy.ext.declarative

Base = sa.ext.declarative.declarative_base()

def get_base():
    return Base

class ResultDB(Base):
    __tablename__ = "results"

    id = sa.Column(sa.BigInteger, primary_key=True)
    clf = sa.Column(sa.String)
    data_aug = sa.Column(sa.String)
    input_dir = sa.Column(sa.String)
    model = sa.Column(sa.String)
    mean_f1 = sa.Column(sa.Float)
    std_f1 = sa.Column(sa.Float)
    mean_accuracy = sa.Column(sa.Float)
    std_accuracy = sa.Column(sa.Float)
    rule = sa.Column(sa.String)

