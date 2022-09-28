import argparse
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Table, Column, Date, Integer, String, Float, MetaData, Sequence, Text
from sqlalchemy.ext.declarative import declarative_base

"""
DATABASE CREATION ROUTINE FOR RECORDING CROSS-VALIDATION RESULTS

# for each dataset need to work out the following:
# AUC for each outcome
# Sens/Spec for each outcome
# Stratified AUC for each outcome
# Brier score for each outcome
# Confidence AUC for each outcome

"""

Base = declarative_base()

class Outcomes(Base):
    __tablename__='Outcomes'

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    outcome = Column(String)
    data = Column(String)
    model = Column(String)
    fold = Column(Float)
    time = Column(String)
    NLL = Column(Float)
    AUC_confidence_a = Column(Float) 
    AUC_confidence_e = Column(Float)
    q25e = Column(Float)
    q25a = Column(Float)
    q75e = Column(Float)
    q75a = Column(Float)
    FRAC_POS = Column(Text)
    MEAN_PRED = Column(Text)
    outside = Column(Float)
    thresh = Column(Float)
    
    def __repr__(self):
        return f'Outcomes {self.outcome}'


metrics = ["AUC", "SEN", "SPEC", "BRIER", "ACC"]
strat = ["full", "q25a", "q75a", "q25e", "q75e", "conf_out"]

for metric in metrics:
    for s in strat:
        setattr(Outcomes, metric + "_" + s, Column(Float))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Objective')
    parser.add_argument('name',
                    help='name for database')
    args = parser.parse_args()

    database = "sqlite:///../../../database/" + args.name + ".db"

    engine = create_engine(database, echo=True)

    Base.metadata.create_all(engine)