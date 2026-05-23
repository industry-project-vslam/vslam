#include <stdio.h>

/*create files*/

// int main(){
//     FILE *fptr;
//     // Create a file
//     fptr = fopen("filename.txt", "w"); // pathing as used in your filesystem
//     // Close the file
//     fclose(fptr);
//     return 0;
// }

/*Closing the file

Did you notice the fclose() function in our example above?

This will close the file when we are done with it.

It is considered as good practice, because it makes sure that:

    Changes are saved properly
    Other programs can use the file (if you want)
    Clean up unnecessary memory space
*/

/*writing to file*/

// int main(){
//     FILE *fptr;
//     // Open a file in writing mode
//     fptr = fopen("filename.txt", "w");
//     // Write some text to the file
//     fprintf(fptr, "Some text");
//     // Close the file
//     fclose(fptr); 
//     return 0;
// }

/*appending to file*/

// int main(){
//     FILE *fptr;
//     // Open a file in append mode
//     fptr = fopen("filename.txt", "a");
//     // Append some text to the file
//     fprintf(fptr, "\nHi everybody!");
//     // Close the file
//     fclose(fptr);
//     return 0;
// }

/*reading the first line*/

// int main(){
//     FILE *fptr;
//     // Open a file in read mode
//     fptr = fopen("filename.txt", "r");
//     // Store the content of the file
//     char myString[100];
//     // Read the content and store it inside myString
//     fgets(myString, 100, fptr);
//     // Print the file content
//     printf("%s", myString);
//     // Close the file
//     fclose(fptr); 
//     return 0;
// }

/*reading every line*/

int main(){
    FILE *fptr;
    // Open a file in read mode
    fptr = fopen("filename.txt", "r");
    // Store 100 characters of the file
    char myString[100];
    // Read the content and print it
    while(fgets(myString, 100, fptr)) {
        printf("%s", myString);
    }
    // Close the file
    fclose(fptr); 
    return 0;
}

/*reading non-existent*/

/*
Good Practice

If you try to open a file for reading that does not exist, the fopen() function will return NULL.

Tip: As a good practice, we can use an if statement to test for NULL, and print some text instead (when the file does not exist):
Example
*/

// int main() {
//     FILE *fptr;
//     // Open a file in read mode
//     fptr = fopen("loremipsum.txt", "r");
//     // Print some text if the file does not exist
//     if(fptr == NULL) {
//         printf("Not able to open the file.");
//     } else {
//         // Close the file if it does exist
//         fclose(fptr);
//     }
    
//     return 0;
// }