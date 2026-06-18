#include <stdio.h>
#include <stdlib.h>

int main(){
    /*Dynamic*/

    int *ptr1, *ptr2;
    ptr1 = malloc(sizeof(*ptr1));
    ptr2 = calloc(1, sizeof(*ptr2));
    
    /*Allocate*/
    
    // Allocate memory
    int *ptr;
    ptr = calloc(4, sizeof(*ptr));

    // Write to the memory
    *ptr = 2;
    ptr[1] = 4;
    ptr[2] = 6;

    // Read from the memory
    printf("%d\n", *ptr);
    printf("%d %d %d", ptr[1], ptr[2], ptr[3]);
    // Dynamic memory does not have its own data type, it is just a sequence of bytes. The data in the memory can be interpreted as a type based on the data type of the pointer.
    // In this example a pointer to four bytes can be interpreted as one int value (4 bytes) or as an array of 4 char values (1 byte each).
    int *ptr1 = malloc(4);
    char *ptr2 = (char*) ptr1;
    ptr1[0] = 1684234849;
    printf("%d is %c %c %c %c", *ptr1, ptr2[0], ptr2[1], ptr2[2], ptr2[3]);

    /*Reallocate*/

    // If the amount of memory you reserved is not enough, you can reallocate it to make it larger.
    // Reallocating reserves a different (usually larger) amount of memory while keeping the data that was stored in it.
    // You can change the size of allocated memory with the realloc() function.
    // The realloc() function takes two parameters:
    //     int *ptr2 = realloc(ptr1, size);
    //
    // The first parameter is a pointer to the memory that is being resized.
    // The second parameter specifies the new size of allocated memory, measured in bytes.
    // The realloc() function tries to resize the memory at ptr1 and return the same memory address.
    // If it cannot resize the memory at the current address then it will allocate memory at a different address and return the new address instead.

    int *ptr1, *ptr2, size;

    // Allocate memory for four integers
    size = 4 * sizeof(*ptr1);
    ptr1 = malloc(size);

    printf("%d bytes allocated at address %p \n", size, ptr1);

    // Resize the memory to hold six integers
    size = 6 * sizeof(*ptr1);
    ptr2 = realloc(ptr1, size);

    printf("%d bytes reallocated at address %p \n", size, ptr2);
    
    // Check for a NULL pointer:
    int *ptr1, *ptr2;

    // Allocate memory
    ptr1 = malloc(4);

    // Attempt to resize the memory
    ptr2 = realloc(ptr1, 8);

    // Check whether realloc is able to resize the memory or not
    if (ptr2 == NULL) {
        // If reallocation fails
        printf("Failed. Unable to resize memory");
    } else {
        // If reallocation is successful
        printf("Success. 8 bytes reallocated at address %p \n", ptr2);
        ptr1 = ptr2;  // Update ptr1 to point to the newly allocated memory
    } 

    /*De-allocate*/
    
    int *ptr;
    ptr = malloc(sizeof(*ptr)); // Allocate memory for one integer
    // If memory cannot be allocated, print a message and end the main() function
    if (ptr == NULL) {
        printf("Unable to allocate memory");
        return 1;  // Exit the program with an error code
    }

    // Set the value of the integer
    *ptr = 20;

    // Print the integer value
    printf("Integer value: %d\n", *ptr);

    // Free allocated memory
    free(ptr);

    // Set the pointer to NULL to prevent it from being accidentally used
    ptr = NULL; 
    return 0;

    /*Memory Leaks

A memory leak happens when dynamic memory is allocated but never freed.

If a memory leak happens in a loop or in a function that is called frequently it could take up too much memory and cause the computer to slow down.

There is a risk of a memory leak if a pointer to dynamic memory is lost before the memory can be freed. This can happen accidentally, so it is important to be careful and keep track of pointers to dynamic memory.

Here are some examples of how a pointer to dynamic memory may be lost.
Example 1

The pointer is overwritten:
int x = 5;
int *ptr;
ptr = calloc(2, sizeof(*ptr));
ptr = &x;

In this example, after the pointer is changed to point at x, the memory allocated by calloc() can no longer be accessed.
Example 2

The pointer exists only inside a function:
void myFunction() {
  int *ptr;
  ptr = malloc(sizeof(*ptr));
}

int main() {
  myFunction();
  printf("The function has ended");
  return 0;
}

In this example, the memory that was allocated inside of the function remains allocated after the function ends but it cannot be accessed anymore. One way to prevent this problem is to free the memory before the function ends.
 Example 3

The pointer gets lost when reallocation fails:
int* ptr;
ptr = malloc(sizeof(*ptr));
ptr = realloc(ptr, 2*sizeof(*ptr));

If realloc() is unable to reallocate memory it will return a pointer to NULL and the original memory will remain reserved.

In this example, if realloc() fails then the NULL pointer is assigned to the ptr variable, overwriting the original memory address so that it cannot be accessed anymore.
*/

}

/*
Structures and Dynamic Memory

You can also use dynamic memory with structures.

This is useful when you don't know how many structs you'll need in advance, or want to save memory by only allocating what's necessary (e.g., in a car dealership program where the number of cars is not fixed).
Allocating Memory for a Struct

You can use the malloc() function to allocate memory for a struct pointer:
Example

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

struct Car {
  char brand[50];
  int year;
};

int main() {
  // Allocate memory for one Car struct
  struct Car *ptr = (struct Car*) malloc(sizeof(struct Car));

  // Check if allocation was successful
  if (ptr == NULL) {
    printf("Memory allocation failed.\n");
    return 1; // Exit the program with an error code
  }

  // Set values
  strcpy(ptr->brand, "Honda");
  ptr->year = 2022;

  // Print values
  printf("Brand: %s\n", ptr->brand);
  printf("Year: %d\n", ptr->year);

  // Free memory
  free(ptr);

  return 0;
}
*/

/*
Using Arrays of Structs

You can also allocate memory for multiple structs at once, like an array:
Example: Allocate memory for 3 cars

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Car {
  char brand[50];
  int year;
};

int main() {
  struct Car *cars = (struct Car*) malloc(3 * sizeof(struct Car));

  if (cars == NULL) {
    printf("Memory allocation failed.\n");
    return 1 // Exit the program with an error code;
  }

  // Fill the data
  strcpy(cars[0].brand, "Ford");
  cars[0].year = 2015;

  strcpy(cars[1].brand, "BMW");
  cars[1].year = 2018;

  strcpy(cars[2].brand, "Volvo");
  cars[2].year = 2023;

  // Print the data
  for (int i = 0; i < 3; i++) {
    printf("%s - %d\n", cars[i].brand, cars[i].year);
  }

  free(cars);
  return 0;
}
*/

/*
Growing Arrays Later with realloc()

If you need more elements later, you can resize your dynamic array with realloc(). This may move the block to a new location and returns a new pointer. Always store the result in a temporary pointer first to avoid losing the original memory if reallocation fails.
Example: Expand an array of structs

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Car {
  char brand[50];
  int year;
};

int main() {
  int count = 2;
  struct Car *cars = (struct Car*) malloc(count * sizeof(struct Car));
  if (cars == NULL) {
    printf("Initial allocation failed.\n");
    return 1;
  }

  // Initialize first 2 cars
  strcpy(cars[0].brand, "Toyota"); cars[0].year = 2010;
  strcpy(cars[1].brand, "Audi");   cars[1].year = 2019;

  // Need one more car -> grow to 3
  int newCount = 3;
  struct Car *tmp = (struct Car*) realloc(cars, newCount * sizeof(struct Car));
  if (tmp == NULL) {
    // 'cars' is still valid here; free it to avoid a leak
    free(cars);
    printf("Reallocation failed.\n");
    return 1;
  }
  cars = tmp;  // use the reallocated block

  // Initialize the new element at index 2
  strcpy(cars[2].brand, "Kia"); 
  cars[2].year = 2022;

  // Print all cars
  for (int i = 0; i < newCount; i++) {
    printf("%s - %d\n", cars[i].brand, cars[i].year);
  }

  free(cars);
  return 0;
}
*/

/*
C Memory Management Example
Real-Life Memory Management Example

To demonstrate a practical example of dynamic memory, we created a program that can make a list of any length.

Regular arrays in C have a fixed length and cannot be changed, but with dynamic memory we can create a list as long as we like:
Example
struct list {
  int *data; // Points to the memory where the list items are stored
  int numItems; // Indicates how many items are currently in the list
  int size; // Indicates how many items fit in the allocated memory
};

void addToList(struct list *myList, int item);

int main() {
  struct list myList;
  int amount;
  int i, j;

  // Create a list and start with enough space for 10 items
  myList.numItems = 0;
  myList.size = 10;
  myList.data = malloc(myList.size * sizeof(int));

  // Find out if memory allocation was successful
  if (myList.data == NULL) {
    printf("Memory allocation failed");
    return 1; // Exit the program with an error code
  }

  // Add any number of items to the list specified by the amount variable
  amount = 44;
  for (i = 0; i < amount; i++) {
    addToList(&myList, i + 1);
  }

  // Display the contents of the list
  for (j = 0; j < myList.numItems; j++) {
    printf("%d ", myList.data[j]);
  }

  // Free the memory when it is no longer needed
  free(myList.data);
  myList.data = NULL;

  return 0;
}

// This function adds an item to a list
void addToList(struct list *myList, int item) {

  // If the list is full then resize the memory to fit 10 more items
  if (myList->numItems == myList->size) {
    int newSize = myList->size + 10;

    // Use a temporary pointer so we don't lose the original on failure
    int *tmp = realloc(myList->data, newSize * sizeof(int));
    if (tmp == NULL) {
      printf("Memory resize failed\n");
      return; // Leave the list unchanged
    }

    // Only update fields after a successful reallocation
    myList->data = tmp;
    myList->size = newSize;
  }

  // Add the item to the end of the list
  myList->data[myList->numItems] = item;
  myList->numItems++;
}

Pointers to structures: This example has a pointer to the structure myList. Because we are using a pointer to the structure instead of the structure itself, we use the arrow syntax (->) to access the structure's members.
Example explained

This example has three parts:

    A structure myList that contains a list's data
    The main() function with the program in it.
    A function addToList() which adds an item to the list

The myList structure

The myList structure contains all of the information about the list, including its contents. It has three members:

    data - A pointer to the dynamic memory which contains the contents of the list
    numItems - Indicates the number of items that list has
    size - Indicates how many items can fit in the allocated memory

We use a structure so that we can easily pass all of this information into a function.

The main() function

The main() function starts by initializing the list with space for 10 items:
// Create a list and start with enough space for 10 items
myList.numItems = 0;
myList.size = 10;
myList.data = malloc(myList.size * sizeof(int));

myList.numItems is set to 0 because the list starts off empty.

myList.size keeps track of how much memory is reserved. We set it to 10 because we will reserve enough memory for 10 items.

We then allocate the memory and store a pointer to it in myList.data.

Then we include error checking to find out if memory allocation was successful:
// Find out if memory allocation was successful
if (myList.data == NULL) {
  printf("Memory allocation failed");
  return 1; // Exit the program with an error code
}

If everything is fine, a loop adds 44 items to the list using the addToList() function:
// Add any number of items to the list specified by the amount variable
amount = 44;
for (i = 0; i < amount; i++) {
  addToList(&myList, i + 1);
}

In the code above, &myList is a pointer to the list and i + 1 is a number that we want to add to the list. We chose i + 1 so that the list would start at 1 instead of 0. You can choose any number to add to the list.

After all of the items have been added to the list, the next loop prints the contents of the list.
// Display the contents of the list
for (j = 0; j < myList.numItems; j++) {
  printf("%d ", myList.data[j]);
}

When we finish printing the list we free the memory to prevent memory leaks.
// Free the memory when it is no longer needed
free(myList.data);
myList.data = NULL;

The addToList() function

Our addToList() function adds an item to the list. It takes two parameters:

void addToList(struct list *myList, int item)

    A pointer to the list.
    The value to be added to the list.

The function first checks if the list is full by comparing the number of items in the list to the size (capacity). If the list is full, it tries to grow the memory to fit 10 more items. We use a temporary pointer with realloc so we don't lose the original block if the resize fails. We only update data and size after a successful resize:

// If the list is full then resize the memory to fit 10 more items
if (myList->numItems == myList->size) {
  int newSize = myList->size + 10;

  // Use a temp pointer so we don't lose the original on failure
  int *tmp = realloc(myList->data, newSize * sizeof(int));
  if (tmp == NULL) {
    printf("Memory resize failed\n");
    return; // Leave the list unchanged
  }

  // Only update fields after a successful reallocation
  myList->data = tmp;
  myList->size = newSize;
}

Finally, the function adds the item to the end of the list. The index at myList->numItems is always at the end of the list because it increases by 1 each time a new item is added.:

// Add the item to the end of the list
myList->data[myList->numItems] = item;
myList->numItems++;

Why do we reserve 10 items at a time?

Optimizing is a balancing act between memory and performance. Even though we may be allocating some memory that we are not using, reallocating memory too frequently can be inefficient. There is a balance between allocating too much memory and allocating memory too frequently.

We chose the number 10 for this example, but it depends on how much data you expect and how often it changes. For example, if we know beforehand that we are going to have exactly 44 items then we can allocate memory for exactly 44 items and only allocate it once.
*/