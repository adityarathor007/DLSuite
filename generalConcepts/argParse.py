import argparse

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

 
    parser.add_argument('--num1',help="Number 1",type=float)   #-- for optional parameters we add '--' (but now while running code we have to provide python file.py --num1 10)
    parser.add_argument('--num2',help="Number 2",type=float)
    parser.add_argument('--operator',help="Operator",default='+')

    args=parser.parse_args()
    print(args)

    result=None

    if args.operator=='+':
        result=args.num1+args.num2
    elif args.operator=='-':
        result=args.num1-args.num2
    elif args.operator=='*':
        result=args.num1*args.num2
    elif args.operator=='pow':
        result=pow(args.num1,args.num2)

    print("Result: ",result)