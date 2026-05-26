/*When a variable is created in C, a memory address is assigned to the variable.

The memory address is the location of where the variable is stored on the computer.

When we assign a value to the variable, it is stored in this memory address.

To access it, use the reference operator (&), and the result represents where the variable is stored:*/

// #include <stdio.h>

// int main(){
//     int myAge = 43;
//     printf("%p", &myAge);
//     return 0;
// }

/*Note: The memory address is in hexadecimal form (0x..).
You will probably not get the same result in your program,
as this depends on where the variable is stored on your computer.

You should also note that &myAge is often called a "pointer".
A pointer basically stores the memory address of a variable as its value.
To print pointer values, we use the %p format specifier.*/

/*A pointer is a variable that stores the memory address of another variable as its value.

A pointer variable points to a data type (like int) of the same type, and is created with the * operator.

The address of the variable you are working with is assigned to the pointer:*/

// #include <stdio.h>

// int main(){
//     int myAge = 43;     // An int variable
//     int* ptr = &myAge;  // A pointer variable, with the name ptr, that stores the address of myAge

//     // Output the value of myAge (43)
//     printf("%d\n", myAge);

//     // Output the memory address of myAge (0x7ffe5367e044)
//     printf("%p\n", &myAge);

//     // Output the memory address of myAge with the pointer (0x7ffe5367e044)
//     printf("%p\n", ptr);
//     return 0;
// }

/*Example explained

Create a pointer variable with the name ptr, that points to an int variable (myAge). Note that the type of the pointer has to match the type of the variable you're working with (int in our example).

Use the & operator to store the memory address of the myAge variable, and assign it to the pointer.

Now, ptr holds the value of myAge's memory address.*/

/*Dereference

In the example above, we used the pointer variable to get the memory address of a variable (used together with the & reference operator).

You can also get the value of the variable the pointer points to, by using the * operator (the dereference operator):*/

// #include <stdio.h>

// int main(){
//     int myAge = 43;     // Variable declaration
//     int* ptr = &myAge;  // Pointer declaration

//     // Reference: Output the memory address of myAge with the pointer (0x7ffe5367e044)
//     printf("%p\n", ptr);

//     // Dereference: Output the value of myAge with the pointer (43)
//     printf("%d\n", *ptr);
//     return 0;
// }

/*Arrays*/

// #include <stdio.h>

// int main(){
//     int myNumbers[4] = {25, 50, 75, 100};
//     int i;

//     for (i = 0; i < 4; i++) {
//         printf("%p\n", &myNumbers[i]);
//     }

//     /*Note that the last number of each of the elements' memory address is different, with an addition of 4.
//     It is because the size of an int type is typically 4 bytes, remember:*/

//     // Create an int variable
//     int myInt;

//     // Get the memory size of an int
//     printf("%zu\n", sizeof(myInt));

//     // Get the size of the myNumbers array
//     printf("%zu\n", sizeof(myNumbers));

//     /*Ok, so what's the relationship between pointers and arrays? Well, in C, the name of an array, is actually a pointer to the first element of the array.
//     Confused? Let's try to understand this better, and use our "memory address example" above again.
//     The memory address of the first element is the same as the name of the array:*/

//     // Get the memory address of the myNumbers array
//     printf("%p\n", myNumbers);

//     // Get the memory address of the first array element
//     printf("%p\n", &myNumbers[0]);

//     /*This basically means that we can work with arrays through pointers!
//     How? Since myNumbers is a pointer to the first element in myNumbers, you can use the * operator to access it:*/

//     // Get the value of the first element in myNumbers
//     printf("%d\n", *myNumbers);

//     /*To access the rest of the elements in myNumbers, you can increment the pointer/array (+1, +2, etc):*/

//     // Get the value of the second element in myNumbers
//     printf("%d\n", *(myNumbers + 1));

//     // Get the value of the third element in myNumbers
//     printf("%d\n", *(myNumbers + 2));

//     /*or loop through it:*/

//     int *ptr = myNumbers;
//     int i;

//     for (i = 0; i < 4; i++) {
//         printf("%d\n", *(ptr + i));
//     }

//     /*It is also possible to change the value of array elements with pointers:*/

//     // Change the value of the first element to 13
//     *myNumbers = 13;

//     // Change the value of the second element to 17
//     *(myNumbers +1) = 17;

//     // Get the value of the first element
//     printf("%d\n", *myNumbers);

//     // Get the value of the second element
//     printf("%d\n", *(myNumbers + 1));

//     return 0;
// }

/*pointer arythmetic
Pointer arithmetic means changing the value of a pointer to make it point to a different element in memory.*/

// #include <stdio.h>

// int main(){
//     int myNumbers[3] = {10, 20, 30};
//     int *p = myNumbers;  // myNumbers[0]

//     printf("%d\n", *p); // 10
//     p++;           // move to myNumbers[1]
//     printf("%d\n", *p); // 20
//     p--;           // back to myNumbers[0]
//     printf("%d\n", *p); // 10

//     p += 2;        // jump to myNumbers[2]
//     printf("%d\n", *p); // 30
//     return 0;
// }

/*Pointer Subtraction (Distance)

You can subtract two pointers that point to elements in the same array to find out how many elements are between them:*/

// #include <stdio.h>

// int main(){
//     int myNumbers[5] = {10, 20, 30, 40, 50};
//     int *start = &myNumbers[1]; // points to 20
//     int *end = &myNumbers[4];   // points to 50

//     printf("%ld\n", end - start); // 3 elements apart 
//     return 0;
// }

/*Pointer Arithmetic Depends on Type

Not all pointers move the same way.

When you add 1 to a pointer, it moves forward by the size of the thing it points to - not just by 1 byte.

For example:

    An int* pointer moves by the size of an integer (usually 4 bytes).
    A char* pointer moves by the size of a character (1 byte).

So if both pointers start at memory address 1000:

    int* → p + 1 would move to address 1004
    char* → p + 1 would move to address 1001

This shows that pointer movement depends on the data type it points to - not on the number you add:*/

#include <stdio.h>

int main(){
    int myNumbers[2] = {1, 2};
    char letters[] = "Hi"; // 'H', 'i', '\0'

    int *pi = myNumbers; // int pointer
    char *pc = letters; // char pointer

    printf("%p\n", (void*)pi);
    printf("%p\n", (void*)(pi + 1)); // moves by sizeof(int) (4 bytes)
    printf("%p\n", (void*)(pi + 2)); // moves by sizeof(int) (4 bytes)

    printf("%p\n", (void*)pc);
    printf("%p\n", (void*)(pc + 1)); // moves by 1 byte
    return 0;
}

/*In the previous chapter, you learned how to loop through an array using *(ptr + i).

Now let's look at another way - by moving the pointer itself inside the loop. Each time the pointer is increased (p++), it moves to the next element in memory:
Example
int myNumbers[4] = {25, 50, 75, 100};
int *p = myNumbers;    // start of array

for (int i = 0; i < 4; i++) {
  printf("%d\n", *p);
  p++; // move to next element
}
  
Here's what happens in each loop:

    *p gives the current element value.
    p++ moves the pointer to the next element in the array.
    No array index (i) is needed - the pointer keeps track of the position.

Tip: This way of looping is common when working directly with memory, since the pointer itself moves through the array instead of using an index number.
*/

/*Common Mistakes to Avoid

    Using the wrong type: Remember that pointer movement depends on its type. An int* moves in 4-byte steps, but a char* moves 1 byte at a time. Mixing them up will point to the wrong memory locations.
    Uninitialized pointers: Always make sure a pointer is pointing to something real before you use it. Using a pointer that doesn't point anywhere can crash your program.
    Going out of bounds: Never move a pointer past the end of an array or before it starts. The only safe "outside" position is one step past the end, and that's only for comparing pointers - not for accessing values.

And be careful; pointers must be handled with care, since it is possible to damage data stored in other memory addresses.*/

/*
Pointer to Pointer

You can also have a pointer that points to another pointer. This is called a pointer to pointer (or "double pointer").

It might sound confusing at first, but it's just one more level of indirection: a pointer that stores the address of another pointer.

Think of it like this: A normal pointer is like a note with an address on it. A pointer to pointer is like another note telling you where that first note is kept.

Note: Pointer to pointer is not something you need to use often as a beginner. However, you might see it in more advanced programs, so it's good to understand what it means and how it works.

Let's look at a simple example to understand how this works:
Example
int myNum = 10;       // normal variable
int *ptr = &myNum;    // pointer to int
int **pptr = &ptr;    // pointer to pointer

printf("myNum = %d\n", myNum);
printf("*ptr = %d\n", *ptr);
printf("**pptr = %d\n", **pptr);

Result:
myNum = 10
*ptr = 10
**pptr = 10

Here's what happens step by step:

    myNum holds the value 10.
    ptr holds the address of myNum.
    pptr holds the address of ptr.
    *ptr gives the value of myNum.
    **pptr also gives the value of myNum, by going through both pointers.

So:

    *ptr = value of myNum
    **pptr = value of myNum through both levels

Changing Values Through a Pointer to Pointer

Since **pptr accesses the original variable, you can use it to change the value of the variable too:
Example
int myNum = 5;
int *ptr = &myNum;
int **pptr = &ptr;

**pptr = 20; // changes myNum

printf("myNum = %d\n", myNum); // prints 20

Summary

    A pointer to pointer stores the address of another pointer.
    *ptr gives the value of a variable.
    **pptr gives the same value by following two levels of indirection.
    They can be useful when passing pointers to functions or working with complex data structures.
*/