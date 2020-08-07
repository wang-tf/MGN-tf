#!/usr/bin/env python3
# coding:utf-8


import data
import loss
import model
from trainer import Trainer

from option import args
import utils.utility as utility

ckpt = utility.checkpoint(args)

loader = data.Data(args)
model = model.Model(args, ckpt)

loss = loss.MGNLoss(args, ckpt) if not args.test_only else None
trainer = Trainer(args, model, loss, loader, ckpt)

for n in range(args.epochs):
  n += 1
  trainer.train(epoch=n)
  if args.test_every!=0 and n % args.test_every == 0:
    trainer.test()
