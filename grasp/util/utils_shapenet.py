import sys
sys.dont_write_bytecode = True

def saveURDF(path, urdfName, meshName, objMass=0.1, x_scale=1, y_scale=1, z_scale=1):

    """
    #* Save URDF file at the specified path with the name. Assume 0.1kg mass and random inertia. Single base link.
    """

    # Write to an URDF file
    f = open(path + urdfName + '.urdf', "w+")

    f.write("<?xml version=\"1.0\" ?>\n")
    f.write("<robot name=\"%s.urdf\">\n" % urdfName)

    f.write("\t<link name=\"baseLink\">\n")
    f.write("\t\t<inertial>\n")
    f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
    f.write("\t\t\t\t<mass value=\"%.1f\"/>\n" % objMass)
    f.write("\t\t\t\t<inertia ixx=\"6e-5\" ixy=\"0\" ixz=\"0\" iyy=\"6e-5\" iyz=\"0\" izz=\"6e-5\"/>\n")
    f.write("\t\t</inertial>\n")

    f.write("\t\t<visual>\n")
    f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
    f.write("\t\t\t<geometry>\n")
    f.write("\t\t\t\t<mesh filename=\"%s\" scale=\"%.2f %.2f %.2f\"/>\n" % (meshName, x_scale, y_scale, z_scale))
    f.write("\t\t\t</geometry>\n")
    f.write("\t\t\t<material name=\"yellow\">\n")
    f.write("\t\t\t\t<color rgba=\"0.98 0.84 0.35 1\"/>\n")
    f.write("\t\t\t</material>\n")
    f.write("\t\t</visual>\n")

    f.write("\t\t<collision>\n")
    f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
    f.write("\t\t\t<geometry>\n")
    f.write("\t\t\t\t<mesh filename=\"%s\" scale=\"%.2f %.2f %.2f\"/>\n" % (meshName, x_scale, y_scale, z_scale))
    f.write("\t\t\t</geometry>\n")
    f.write("\t\t</collision>\n")
    f.write("\t</link>\n")
    f.write("</robot>\n")

    f.close()