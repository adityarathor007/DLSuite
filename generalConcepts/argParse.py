import argparse

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description="Calculator")

 
    parser.add_argument('--num1',help="Number 1",type=float)   #-- for optional parameters we add '--' (but now while running code we have to provide python file.py --num1 10)
    parser.add_argument('--num2',help="Number 2",type=float)
    parser.add_argument('--operator',help="Operator",default='+')   

    args=parser.parse_args()
    print(args)
    print(args.num1)



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


    # to use in the command line 
    # python argParse.py --num1 2 --num2 4 --operator 
    # python argParse.py -h (to get to know what option are available)