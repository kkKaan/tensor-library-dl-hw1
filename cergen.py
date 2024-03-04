import random
import math
from typing import Union

def cekirdek(sayi: int):
    """
    Sets the seed for random number generation
    """
    random.seed(sayi)

def generate_nested_list_dogal(shape, aralik, depth=0):
    """
    Generates a nested list of integers with the specified shape and range.
    """
    if depth == len(shape) - 1:
        return [random.randint(aralik[0], aralik[1]) for _ in range(shape[depth])]
    else:
        return [generate_nested_list_dogal(shape, aralik, depth + 1) for _ in range(shape[depth])]

def rastgele_dogal(boyut, aralik=(0,100), dagilim='uniform'):
    """
    Generates data of specified dimensions with random integer values and returns a gergen object.

    Parameters:
    boyut (tuple): Shape of the desired data.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0,100), which implies a default range.
    dagilim (string, optional): Distribution of random values ('uniform'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random integer values.
    """
    if dagilim != 'uniform':
        raise ValueError('Invalid distribution.')
    
    veri = generate_nested_list_dogal(boyut, aralik)
    return gergen(veri)

def generate_nested_list_gercek(shape, aralik, depth=0):
    """
    Generates a nested list of floating-point numbers with the specified shape and range.
    """
    if depth == len(shape) - 1:
        return [random.random() * (aralik[1] - aralik[0]) + aralik[0] for _ in range(shape[depth])]
    else:
        return [generate_nested_list_gercek(shape, aralik, depth + 1) for _ in range(shape[depth])]

def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    """
    Generates a gergen of specified dimensions with random floating-point values.

    Parameters:
    boyut (tuple): Shape of the desired gergen.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0.0, 1.0) for uniform distribution.
    dagilim (string, optional): Distribution of random value ('uniform'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random floating-point values.
    """
    if dagilim != 'uniform':
        raise ValueError('Invalid distribution.')
    
    veri = generate_nested_list_gercek(boyut, aralik)
    return gergen(veri)

class Operation:
    def __call__(self, *operands):
        """
        Makes an instance of the Operation class callable.
        Stores operands and initializes outputs to None.
        Invokes the forward pass of the operation with given operands.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the forward pass of the operation.
        """
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError
    
class gergen:
    __veri = None # A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None # Dimensions of the derivative (Shape)

    def __init__(self, veri=None):
        """
        The constructor for the 'gergen' class.
        
        This method initializes a new instance of a gergen object. The gergen can be
        initialized with data if provided; otherwise, it defaults to None, representing
        an empty tensor.
        
        Parameters:
        veri (int/float, list, list of lists, optional): A nested list of numbers that represents the
        gergen data. The outer list contains rows, and each inner list contains the
        elements of each row. If 'veri' is None, the tensor is initialized without data.
        
        Example:
        To create a tensor with data, pass a nested list:
        tensor = gergen([[1, 2, 3], [4, 5, 6]])
        
        To create an empty tensor, simply instantiate the class without arguments:
        empty_tensor = gergen()
        """
        self.__veri = self.__validate_veri(veri)
        self.__boyut = self.__calculate_boyut(veri)
        self.D = None  # Placeholder for transpose, which needs its own implementation

    def __validate_veri(self, veri):
        """
        Recursively checks if the input data (veri) is valid (i.e., all sublists are of equal length).
        Returns the validated data or raises a ValueError if the data is invalid.
        """
        if veri is None:
            return None
        elif isinstance(veri, (int, float)):
            return veri
        elif isinstance(veri, list):
            if not veri:
                return veri
            if all(isinstance(item, (int, float)) for item in veri):
                return veri
            elif all(isinstance(item, list) for item in veri):
                length = len(veri[0])
                for sublist in veri:
                    if len(sublist) != length or not all(isinstance(item, (int, float, list)) for item in sublist):
                        raise ValueError('All sublists must be of equal length and contain only numbers or sublists.')
                    self.__validate_veri(sublist)
                return veri
            else:
                raise ValueError('Invalid data type in veri.')
        else:
            raise ValueError('veri must be a number, a list of numbers, or a nested list of numbers.')

    def __calculate_boyut(self, veri):
        """
        Recursively calculates the dimensions (shape) of the tensor based on the input data (veri).
        Returns a tuple representing the dimensions of the tensor.
        """
        if veri is None:
            return None
        elif isinstance(veri, (int, float)):
            return ()
        else:
            boyut = []
            while isinstance(veri, list):
                boyut.append(len(veri))
                veri = veri[0] if veri else []
            # Check if only 1 element in the tensor
            if len(boyut) == 1:
                boyut.append(1)
            return tuple(boyut)

    def __getitem__(self, index):
        """
        Enables indexing the gergen, allowing access to specific elements, rows, columns, 
        or sub-tensors using standard Python indexing syntax. This method will be called 
        when an index is used on an instance of the gergen class, such as g[0] for a gergen g.
        """
        if self.__veri is None:
            raise ValueError("Cannot index an empty gergen.")

        if not isinstance(index, tuple):
            index = (index,)  # Make single indexes into a tuple for uniform handling
        
        current = self.__veri
        for idx in index:
            if not isinstance(current, list):
                raise IndexError("Indexing beyond the tensor dimensions")
            if idx < 0:
                idx += len(current)
            if idx < 0 or idx >= len(current):
                raise IndexError("Index out of range")
            current = current[idx]
        return current

    def __str__(self):
        """
        Generates a string representation of the gergen. When
        print(g1) is invoked on a gergen instance g1, this method is automatically called,
        resulting in the display of the tensor’s boyut followed by its veri.
        """
        if self.__veri is None:
            return 'Empty gergen'
        elif isinstance(self.__veri, (int, float)):
            return f'0 boyutlu skaler gergen:\n{self.__veri}'
        else:
            # Helper function to format nested lists correctly
            def format_nested_list(veri):
                if all(isinstance(item, list) for item in veri):  # Multi-dimensional
                    # Format each row with brackets and join with newline in between
                    formatted_rows = ['[' + ', '.join(map(str, row)) + ']' for row in veri]
                    return '[' + '\n'.join(formatted_rows) + ']'
                else:  # Single-dimensional
                    return '[' + ', '.join(map(str, veri)) + ']'

            # Determine if we're dealing with a multi-dimensional tensor or a vector of any size
            boyut = ''
            for i in range(len(self.__boyut)):
                if i != len(self.__boyut) - 1:
                    boyut += f'{self.__boyut[i]}x'
                else:
                    boyut += f'{self.__boyut[i]}'

            # Use the helper function to format the tensor data
            formatted_data = format_nested_list(self.__veri)

            return f'{boyut} boyutlu gergen:\n{formatted_data}'

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        This method facilitates the multiplication of the gergen either with another gergen instance for
        element-wise multiplication, or with a scalar (int/float), yielding a new gergen object
        as the result. The other parameter is permitted to be a gergen, an integer, or
        a floating-point number. Error handling is incorporated to manage cases where the
        other parameter is neither a gergen object nor a numerical scalar. If the dimensions
        of two gergen instances do not align for element-wise multiplication, or if an
        incompatible type is provided for other, a TypeError or ValueError is raised.
        """
        if self.__veri is None:
            raise ValueError("Cannot multiply an empty gergen.")
        
        if isinstance(other, (int, float)):
            if isinstance(self.__veri, (int, float)):
                # Multiply two scalars
                new_data = self.__veri * other
                return gergen(new_data)
            else:
                # Multiply each element of the gergen of any size by the scalar
                def scalar_mult(data, scalar):
                    if isinstance(data, list):
                        return [scalar_mult(subdata, scalar) for subdata in data]
                    else:
                        return data * scalar
                new_data = scalar_mult(self.__veri, other)
                return gergen(new_data)
        elif isinstance(other, gergen):
            # Element-wise multiplication of two gergen objects
            if self.__boyut != other.__boyut:
                raise ValueError("Cannot multiply gergens with different dimensions.")
            else:
                # Multiply each element of the gergen of any size by the corresponding element of the other gergen
                def elementwise_mult(data1, data2):
                    if isinstance(data1, list) and isinstance(data2, list):
                        return [elementwise_mult(subdata1, subdata2) for subdata1, subdata2 in zip(data1, data2)]
                    else:
                        return data1 * data2
                new_data = elementwise_mult(self.__veri, other.__veri)
                return gergen(new_data)          

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        This method implements division for the gergen, facilitating element-wise division by a scalar
        (an integer or a float), and encapsulates the result in a new gergen instance. True division is 
        employed, ensuring that the result is always a floating-point number, consistent
        with Python 3.x division behavior, even if both operands are integers. Error handling
        mechanism ahould check potential issues: if other is zero, a ZeroDivisionError is
        raised to prevent division by zero. Additionally, if other is not a scalar type (int or
        float), a TypeError is raised to enforce the type requirement for the scalar divisor.
        """
         # Check for division by zero
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        
        # Case where the divisor is a scalar (int or float)
        if isinstance(other, (int, float)):
            # Recursive function for scalar division
            def scalar_div(data, scalar):
                if isinstance(data, list):
                    # If the current item is a list, recur for each item
                    return [scalar_div(subdata, scalar) for subdata in data]
                else:
                    # Base case: data is not a list (i.e., an actual number)
                    return data / scalar  # Perform true division

            # Apply scalar division to the entire nested list structure
            new_data = scalar_div(self.__veri, other)
            return gergen(new_data)
        elif isinstance(other, gergen):
            # Element-wise division of two gergen objects
            if self.__boyut != other.__boyut:
                raise ValueError("Cannot divide gergens with different dimensions.")
            else:
                # Recursive function for element-wise division
                def elementwise_div(data1, data2):
                    if isinstance(data1, list) and isinstance(data2, list):
                        # If both items are lists, recur for each item
                        return [elementwise_div(subdata1, subdata2) for subdata1, subdata2 in zip(data1, data2)]
                    else:
                        # Base case: data1 and data2 are not lists (i.e., actual numbers)
                        return data1 / data2  # Perform true division
                    
                # Apply element-wise division to the entire nested list structure
                new_data = elementwise_div(self.__veri, other.__veri)
                return gergen(new_data)
        else:
            raise TypeError("Divisor must be a scalar or a gergen object.")    
        
    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the addition operation for gergen objects.
        Called when a gergen object is added to another, using the '+' operator.
        The operation is element-wise.
        """
        pass

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects.
        Called when a gergen object is subtracted from another, using the '-' operator.
        The operation is element-wise.
        """
        pass

    def uzunluk(self):
        """
        Returns the total number of elements in the gergen
        """
        pass

    def boyut(self):
        """
        Returns the shape of the gergen
        """
        pass

    def devrik(self):
    # Returns the transpose of gergen
        pass

    def sin(self):
    # Calculates the sine of each element in the given `gergen`.
        pass

    def cos(self):
    # Calculates the cosine of each element in the given `gergen`.
        pass

    def tan(self):
    # Calculates the tangent of each element in the given `gergen`.
        pass

    def us(self, n: int):
    # Raises each element of the gergen object to the power 'n'. This is an element-wise operation.
        pass

    def log(self):
    # Applies the logarithm function to each element of the gergen object, using the base 10.
        pass

    def ln(self):
    # Applies the natural logarithm function to each element of the gergen object.
        pass

    def L1(self):
    # Calculates and returns the L1 norm
        pass

    def L2(self):
    # Calculates and returns the L2 norm
        pass

    def Lp(self, p):
    # Calculates and returns the Lp norm, where p should be positive integer
        pass

    def listeye(self):
    # Converts the gergen object into a list or a nested list, depending on its dimensions.
        pass

    def duzlestir(self):
    # Converts the gergen object's multi-dimensional structure into a 1D structure, effectively 'flattening' the object.
        pass

    def boyutlandir(self, yeni_boyut):
    # Reshapes the gergen object to a new shape 'yeni_boyut', which is specified as a tuple.
        pass

    def ic_carpim(self, other):
    # Calculates the inner (dot) product of this gergen object with another.
        pass

    def dis_carpim(self, other):
    #Calculates the outer product of this gergen object with another.
        pass
    
    def topla(self, eksen=None):
    #Sums up the elements of the gergen object, optionally along a specified axis 'eksen'.
        pass

    def ortalama(self, eksen=None):
    #Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        pass

def main():
    # a main function to test the functions

    cekirdek(2)
    g = rastgele_dogal((3, 3, 2, 4))

    # test the random number generators

    # g1 = rastgele_dogal((3, 3))
    # g1 = rastgele_gercek((3, 3, 3))
    # print(g1)
    # print(gergen())
    # print(g1)
    # print(g1[0][-1])

    # scaler gergens

    # g3 = gergen(2)
    # print(g3)
    # print(g3.D)

    # test the __getitem__ method

    # print(g)
    # print(g[0])
    # print(g[0, 1]) # = print(g[0][1])
    # print(g[0, 2, 1])

    # test the __str__ method

    # print(gergen())
    # print(gergen(2))
    # print(gergen([[1, 2, 3]]))
    # print(gergen([[1, 2, 3], [4, 5, 6]]))

    # print(g)
    # print()
    # print(g[0])
    # print()
    # print(g[0, 1])
    # print()
    # print(g[0, 2, 1])
    # print()
    # print(g[0, 2, -1, 3])

    # test the __mul__ method

    # g1 = gergen([[1, 2, 3], [4, 5, 6]])
    # gs1 = g1 * 2
    # # gss1 = g1 * gergen(2) # is not working
    # g2 = gergen([[7, 8, 9], [10, 11, 12]])
    # gs2 = g2 * -3 # (-3 * g2) is not working
    # # gss2 = gergen(-3) * g2 # is not working
    # g3 = g1 * g2
    # g4 = g1 * g2 * 2 # 
    # g5 = g1 * g2 * g3 * 2 # 
    # g1d1 = gergen(4)
    # g1d2 = gergen(3)
    # g1dm = g1d1 * g1d2

    # print(g3) # [[7, 16, 27], [40, 55, 72]]
    # print()
    # print(gs1) 
    # print()
    # print(gs2)
    # print()
    # print(g4) # [[14, 32, 54], [80, 110, 144]]
    # print()
    # print(g5) 
    # print()
    # print(g1dm)
    # print()
    # print(g1dm * g3)

    # test the __truediv__ method

    # g1 = gergen([[1, 2, 3], [4, 5, 6]])
    # z1 = 0
    # gz1 = g1 / z1
    # print(gz1)

    # g2 = g1 * 2
    # z2 = 2
    # gz2 = g2 / z2
    # print(g2)
    # print()
    # print(gz2)

    # g3 = g1 / g1
    # print(g3)

    # g4 = gergen([[7, 8, 9], [10, 11, 12]])
    # print(g4 * 4 / g4)

    g5 = gergen([6])
    print(g5 / 3)
    g6 = gergen([6, 7])
    # print(g5 / g6) # ValueError: Cannot divide gergens with different dimensions.
    g7 = gergen([3])
    print(g5 / g7)

    # test the __add__ method





          


if __name__ == "__main__":
    main()