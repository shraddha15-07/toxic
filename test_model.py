import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from predictor import ToxicityPredictor

pred = ToxicityPredictor(model_dir='./model')
tests = [
    'I hate you so much',
    'i hate you so much',
    'This is a great article',
    'I love this'
]

print("Thresholds:", pred.thresholds)
for t in tests:
    res = pred.predict(t)
    print(f"'{t}':")
    print(f"  Toxic prob: {res['probabilities']['toxic']:.4f}")
    print(f"  Triggered: {res['labels_triggered']}")
    print()
